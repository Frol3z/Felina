#include "Renderer.hpp"

#include "Application.hpp"
#include "Mesh.hpp"
#include "Scene.hpp"
#include "Window.hpp"
#include "Device.hpp"
#include "Swapchain.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"
#include "GBuffer.hpp"
#include "PipelineBuilder.hpp"
#include "ResourceManager.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <stb_image.h>

#include <iostream>
#include <filesystem>
#include <cassert>

namespace Felina 
{
    // Validation layers
    #ifdef NDEBUG
        constexpr bool enableValidationLayers = false;
    #else
        constexpr bool enableValidationLayers = true;
    #endif

    const std::vector validationLayers =
    {
        "VK_LAYER_KHRONOS_validation",  // Khronos validation layers
        "VK_LAYER_LUNARG_monitor"       // FPS monitoring layer
    };

	Renderer::Renderer(Application& app, const Window& window, const Scene& scene)
        : m_app(app), m_window(window), m_scene(scene)
    {
        // Vulkan backend initialization
        CreateInstance();
        CreateSurface();
        CreateDevice();
        CreateSwapchain();
        CreateDescriptorPool();
        CreateGBuffer();
        CreateDescriptorSetLayouts();
        CreatePushConstant();
        CreatePipeline();
        CreateCommandPool();
        CreateCommandBuffer();
        CreateSamplers();
        CreateUniformBuffers();
        AllocateDescriptorSets();
        CreateSyncObjects();
    }

    Renderer::~Renderer()
    {
        // Destroying buffers explicitly before freeing up memory
        for (auto& buffer : m_cameraUBOs) {
            buffer.reset();
        }
        for (auto& buffer : m_objectSSBOs) {
            buffer.reset();
        }
    }

    void Renderer::DrawFrame()
    {
        // CPU will wait until the GPU finishes rendering the previous frame (corresponding to the same swapchain image index)
        while (vk::Result::eTimeout == m_device->GetDevice().waitForFences(*m_inFlightFences[m_currentFrame], vk::True, UINT64_MAX));

        // Wait for the presentation to release the semaphore
        m_device->GetPresentQueue().waitIdle();

        // Check if window has been resized/minimize before trying to acquire next image
        if (m_app.IsFramebufferResized()) {
            UpdateOnFramebufferResized();
            return;
        }

        // Acquire next image index and signal the semaphore when done
        auto resultStruct = m_swapchain->AcquireNextImage(*m_imageAvailableSemaphores[m_currentFrame]);
        auto result = resultStruct.result; // vulkan.hpp changed their return value from std::pair
        auto imageIndex = resultStruct.value; // to a custom struct

        // Check if the surface is still compatible with the swapchain or if it was resized/minimized
        // !!! Checking m_IsFramebufferResized guarantees prevents presentation to an invalid surface
        if (result == vk::Result::eErrorOutOfDateKHR || m_app.IsFramebufferResized()) {
            UpdateOnFramebufferResized();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swapchain image!");
        }

        SetupFrameData();

        // Record command buffer and reset draw fence
        m_device->GetDevice().resetFences(*m_inFlightFences[m_currentFrame]);
        RecordCommandBuffer(imageIndex);

        // Submit commands to the queue
        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*m_imageAvailableSemaphores[m_currentFrame],
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*m_commandBuffers[m_currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*m_renderFinishedSemaphores[m_currentFrame]
        };
        m_device->GetGraphicsQueue().submit(submitInfo, *m_inFlightFences[m_currentFrame]);

        // Present to the screen
        const vk::PresentInfoKHR presentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*m_renderFinishedSemaphores[m_currentFrame],
            .swapchainCount = 1,
            .pSwapchains = &m_swapchain->GetHandle(),
            .pImageIndices = &imageIndex
        };
        result = m_device->GetPresentQueue().presentKHR(presentInfoKHR);

        // Check again if presentation fails because the surface is now incompatible
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || m_app.IsFramebufferResized()) {
            UpdateOnFramebufferResized();
            return;
        }
        else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swapchain image!");
        }

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void Renderer::WaitIdle()
    {
        m_device->GetDevice().waitIdle();
    }

    void Renderer::LoadMesh(Mesh& mesh)
    {
        mesh.Load(*m_device);
    }

    void Renderer::LoadTexture(const Texture& texture, const void* rawImageData, size_t rawImageSize)
    {
        // When updating image data we recommend the use of staging buffers,
        // rather than staging images for our hardware
        // From here: https://developer.nvidia.com/vulkan-memory-management

        // Create and fill staging buffer
        vk::DeviceSize size = static_cast<vk::DeviceSize>(rawImageSize);
        vk::BufferCreateInfo stagingBufferCreateInfo
        {
            .size = size,
            .usage = vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
        };
        // See https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
        // Staging copy for upload section
        VmaAllocationCreateInfo allocCreateInfo{ 
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                   | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_CPU_TO_GPU
        };
        Buffer stagingBuffer{ m_device->GetAllocator(), stagingBufferCreateInfo, allocCreateInfo };
        
        stagingBuffer.LoadData(rawImageData, rawImageSize);
        m_device->CopyBufferToImage(stagingBuffer, texture, size);
    }

    void Renderer::LoadSkybox(const std::filesystem::path& folderPath)
    {
        constexpr int requiredComponents = 4;
        constexpr int faceCount = 6;

        std::array<unsigned char*, faceCount> faces { };
        int width{ 0 };
        int height{ 0 };

        // Load the faces images
        size_t i { 0 };
        for (auto const& dirEntry : std::filesystem::directory_iterator{ folderPath })
        {
            const std::string& path = dirEntry.path().string();
            LOG("[Renderer] Loading " + path + "...");

            int w, h, c;
            faces[i] = stbi_load(path.c_str(), &w, &h, &c, requiredComponents);
            if (!faces[i])
                throw std::runtime_error("[Renderer] Failed to load skybox face #" + std::to_string(i));

            if(i == 0)
            {
                width = w;
                height = h;
            }
            else
            {
                assert(width == w && height == h);
            }
            i++;
        }
        assert(i == 6);

        // Pack the data into a single buffer
        size_t faceSize = width * height * requiredComponents;
        size_t cubeMapSize = faceSize * faceCount;
        std::vector<unsigned char> cubeMapData(cubeMapSize);
        for (size_t face = 0; face < 6; face++)
        {
            memcpy(
                cubeMapData.data() + face * faceSize, // Dst
                faces[face],                          // Src
                faceSize                              // Size
            );
        }

        // Create texture
        vk::ImageCreateInfo imageInfo{
            .flags          = vk::ImageCreateFlagBits::eCubeCompatible, // !
            .imageType      = vk::ImageType::e2D,
            .format         = vk::Format::eR8G8B8A8Srgb,
            .extent         = vk::Extent3D{ static_cast<uint32_t>(width),static_cast<uint32_t>(height), 1 },
            .mipLevels      = 1,
            .arrayLayers    = 6, // !
            .samples        = vk::SampleCountFlagBits::e1,
            .tiling         = vk::ImageTiling::eOptimal,
            .usage          = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode    = vk::SharingMode::eExclusive,
            .initialLayout  = vk::ImageLayout::eUndefined
        };
        VmaAllocationCreateInfo allocInfo = { .usage = VMA_MEMORY_USAGE_AUTO };
        std::unique_ptr<Texture> texture = std::make_unique<Texture>(*m_device, imageInfo, allocInfo);

        // Load texture
        // TODO: Renderer calling RM calling Renderer again, I should fix this weird process
        ResourceManager::GetInstance().LoadTexture(std::move(texture), "Skybox", cubeMapData.data(), cubeMapSize, *this);
        // TODO: free stb_image_data
    }

    void Renderer::UpdateDescriptorSets()
    {
        // Bind the textures and samplers to their descriptor set (shared between frames)
       {
           // Samplers
           std::array<vk::DescriptorImageInfo, MAX_SAMPLERS> samplerInfos;
           for (size_t i = 0; i < samplerInfos.size(); i++)
           {
               samplerInfos[i] = {
                .sampler = m_samplers[i].value(),
                .imageView = nullptr,
                .imageLayout = vk::ImageLayout::eUndefined
               };
           }

           // Textures (array + skybox cubemap)
           auto& rm = ResourceManager::GetInstance();
           const auto& textures = rm.GetTextures();
           std::array<vk::DescriptorImageInfo, MAX_TEXTURES> imageInfos;
           vk::DescriptorImageInfo skyboxInfo; // Skybox doesn't count in MAX_TEXTURES
           size_t i = 0;
           for (const auto& [id, resource] : textures)
           {
               // Check wether it is the skybox
               if (resource.resource->IsCubemap())
               {
                   skyboxInfo = {
                       .sampler = nullptr,
                       .imageView = resource.resource->GetImageView(),
                       .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                   };
               }
               else
               {
                   imageInfos[i] = {
                       .sampler = nullptr,
                       .imageView = resource.resource->GetImageView(),
                       .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                   };
                   ++i;
               }
           }

           // Descriptor writes
           std::array<vk::WriteDescriptorSet, 3> writes;
           writes[0] = {
             .dstSet = m_textureDescriptorSets,
             .dstBinding = 0, // samplers
             .dstArrayElement = 0,
             .descriptorCount = static_cast<uint32_t>(samplerInfos.size()),
             .descriptorType = vk::DescriptorType::eSampler,
             .pImageInfo = samplerInfos.data()
           };
           writes[1] = {
                .dstSet = m_textureDescriptorSets,
                .dstBinding = 1, // texture
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(imageInfos.size()),
                .descriptorType = vk::DescriptorType::eSampledImage,
                .pImageInfo = imageInfos.data()
           };
           writes[2] = {
             .dstSet = m_textureDescriptorSets,
             .dstBinding = 2, // skybox
             .dstArrayElement = 0,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &skyboxInfo
           };
           m_device->GetDevice().updateDescriptorSets(writes, {});
       }

       // Bind the buffers to the corresponding set (per frame in flight)
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            // Camera descriptor set
            vk::DescriptorBufferInfo cameraUBOInfo{
                .buffer = m_cameraUBOs[i]->GetHandle(),
                .offset = 0,
                .range = sizeof(CameraData)
            };
            vk::WriteDescriptorSet cameraWrite{
                .dstSet = m_cameraDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &cameraUBOInfo
            };
            m_device->GetDevice().updateDescriptorSets(cameraWrite, {});

            // Lights descriptor set
            vk::DescriptorBufferInfo lightUBOInfo{
                .buffer = m_lightUBOs[i]->GetHandle(),
                .offset = 0,
                .range = sizeof(LightsUBO)
            };
            vk::WriteDescriptorSet lightWrite{
                .dstSet = m_lightDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &lightUBOInfo
            };
            m_device->GetDevice().updateDescriptorSets(lightWrite, {});

            // Object descriptor set
            vk::DescriptorBufferInfo objectSSBOInfo{
                .buffer = m_objectSSBOs[i]->GetHandle(),
                .offset = 0,
                .range = sizeof(ObjectData) * MAX_OBJECTS
            };
            vk::WriteDescriptorSet objectWrite{
                .dstSet = m_objectDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &objectSSBOInfo
            };
            m_device->GetDevice().updateDescriptorSets(objectWrite, {});

            // Material descriptor set
            vk::DescriptorBufferInfo materialSSBOInfo{
                .buffer = m_materialSSBOs[i]->GetHandle(),
                .offset = 0,
                .range = sizeof(MaterialData) * MAX_MATERIALS
            };
            vk::WriteDescriptorSet materialWrite{
                .dstSet = m_materialDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &materialSSBOInfo
            };
            m_device->GetDevice().updateDescriptorSets(materialWrite, {});
        }
    }

    const Device& Renderer::GetDevice() const
    {
        return *m_device;
    }

    ImGui_ImplVulkan_InitInfo Renderer::GetImGuiInitInfo()
    {
        vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.colorAttachmentCount = 1;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchain->GetSurfaceFormat().format;
        pipelineRenderingCreateInfo.depthAttachmentFormat = vk::Format::eUndefined;
        pipelineRenderingCreateInfo.stencilAttachmentFormat = vk::Format::eUndefined;

        // Create shader module
        auto shaderPath = std::filesystem::current_path() / "./shaders/imgui_custom.vert.spv";
        m_imGuiCustomVertShaderCode = ReadFile(shaderPath.string());
        m_imGuiCustomVertShaderModuleCreateInfo = {
            .codeSize = m_imGuiCustomVertShaderCode.size() * sizeof(char),
            .pCode = reinterpret_cast<const uint32_t*>(m_imGuiCustomVertShaderCode.data())
        };

        ImGui_ImplVulkan_InitInfo vkInitInfo = {};
        vkInitInfo.ApiVersion = vk::ApiVersion14;
        vkInitInfo.Instance = *m_instance;
        vkInitInfo.PhysicalDevice = *m_device->GetPhysicalDevice();
        vkInitInfo.Device = *m_device->GetDevice();
        vkInitInfo.QueueFamily = m_device->GetGraphicsQueueFamilyIndex();
        vkInitInfo.Queue = *m_device->GetGraphicsQueue();
        vkInitInfo.DescriptorPool = VK_NULL_HANDLE;
        vkInitInfo.DescriptorPoolSize = 1000; // ImGui backend will allocate the descriptor pool
        vkInitInfo.MinImageCount = MAX_FRAMES_IN_FLIGHT;
        vkInitInfo.ImageCount = static_cast<uint32_t>(m_swapchain->GetImages().size());
        vkInitInfo.PipelineInfoMain.RenderPass = VK_NULL_HANDLE;
        vkInitInfo.PipelineInfoMain.Subpass = 0;
        vkInitInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        vkInitInfo.PipelineInfoMain.PipelineRenderingCreateInfo = *&pipelineRenderingCreateInfo;
        vkInitInfo.UseDynamicRendering = true;
        vkInitInfo.CustomShaderVertCreateInfo = *&m_imGuiCustomVertShaderModuleCreateInfo;
        vkInitInfo.CustomShaderFragCreateInfo = {};
        return vkInitInfo;
    }

    void Renderer::DrawImGuiFrame(ImDrawData* drawData)
    {
        ImGui_ImplVulkan_RenderDrawData(drawData, *m_commandBuffers[m_currentFrame]);
    }

    static void UpdateObject(const Object& obj, glm::mat4 parentModelMatrix, 
        std::vector<Renderer::ObjectData>& objectDatas, std::vector<Renderer::LightData>& lightDatas
    )
    {
        // Add current object data
        glm::mat4 modelMatrix = parentModelMatrix * obj.GetModelMatrix();
        if (obj.GetMeshes().size())
        {
            objectDatas.emplace_back(Renderer::ObjectData{
                .model = modelMatrix,
                .normal = glm::transpose(glm::inverse(glm::mat3(modelMatrix)))
                // .normal = glm::mat3(modelMatrix) (uniform scaling ONLY assumption)
            });
        }

        // Process light data
        std::optional<Light> lightObjData = obj.GetLightData();
        if (lightObjData.has_value())
        {
            const Light& light = lightObjData.value();
            glm::vec4 defaultLightDir = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f); // see KHR_punctual_lights glTF extension
            glm::vec3 lightPos = modelMatrix * glm::vec4(obj.GetPosition(), 1.0f);
            glm::vec3 lightDir = glm::normalize(modelMatrix * defaultLightDir);
            
            Renderer::LightData ld{}; // zero-initialize everything, including padding

            ld.type = static_cast<uint32_t>(light.type);
            ld.color = light.color;
            ld.position = lightPos;
            ld.direction = lightDir;
            ld.intensity = light.intensity;
            ld.range = light.range;
            ld.innerConeAngle = light.innerConeAngle;
            ld.outerConeAngle = light.outerConeAngle;

            lightDatas.push_back(ld);
        }

        // Iterate through its children
        for (const auto& childPtr : obj.GetChildren())
        {
            const Object& child = *childPtr;
            UpdateObject(child, modelMatrix, objectDatas, lightDatas);
        }
    }

    void Renderer::SetupFrameData()
    {   
        // Update camera data
        CameraData cameraData{};
        cameraData.position = m_scene.GetCamera().GetPosition();
        cameraData.view = m_scene.GetCamera().GetViewMatrix();
        cameraData.proj = m_scene.GetCamera().GetProjectionMatrix();
        cameraData.invViewProj = m_scene.GetCamera().GetInvViewProj();
        m_cameraUBOs[m_currentFrame]->LoadData(&cameraData, sizeof(cameraData));

        // Fill the object data storage buffer and lights data uniform buffer
        const std::vector<std::unique_ptr<Object>>& objects = m_scene.GetObjects();
        std::vector<ObjectData> objectDatas;
        std::vector<LightData> lightDatas;
        lightDatas.reserve(MAX_LIGHTS);

        // Scene hierarchy traversal
        for (const auto& objPtr : objects)
        {
            const Object& obj = *objPtr;
            UpdateObject(obj, glm::mat4(1.0f), objectDatas, lightDatas);
        }

        // Create LightsUBO
        LightsUBO lightsUBO{};
        lightsUBO.lightsCount = lightDatas.size();
        std::copy(lightDatas.begin(), lightDatas.end(), lightsUBO.lights.begin());

        m_objectSSBOs[m_currentFrame]->LoadData(objectDatas.data(), objectDatas.size() * sizeof(ObjectData));
        m_lightUBOs[m_currentFrame]->LoadData(&lightsUBO, sizeof(LightsUBO));
        
        // TODO: optimize by avoid doing these iterations each frame if not needed
        // Iterate through texture to fill up the look-up table
        auto& rm = ResourceManager::GetInstance();
        auto& texturesMapping = m_textureIDToArrayID[m_currentFrame];
        uint32_t index = 0;
        for (const auto& [id, res] : rm.GetTextures())
        {
            if (res.resource->IsCubemap())
                continue;
            texturesMapping[id] = index++;
        }

        // Fill the material data storage buffer
        std::vector<MaterialData> materialDatas;
        auto& materialsMapping = m_materialIDToSSBOID[m_currentFrame];
        index = 0;
        for (const auto& [id, res] : ResourceManager::GetInstance().GetMaterials())
        {
            // Get raw pointer to the material
            const Material* mat = res.resource.get();

            // Fill the MaterialData struct
            MaterialData matData{};
            matData.baseColor = mat->GetBaseColor();
            matData.materialInfo = mat->GetMetallicRoughness();

            // If the texture is defined use the mapping to get the
            // correct texture index, else -1
            TextureID texId = mat->GetBaseColorTexture();
            matData.baseColorTex = (texId != -1) ? texturesMapping[texId] : texId;

            texId = mat->GetMetallicRoughnessTexture();
            matData.materialInfoTex = (texId != -1) ? texturesMapping[texId] : texId;
            
            materialDatas.push_back(matData);

            // Keep track of the storage buffer IDs
            materialsMapping[id] = index++;
        }
        m_materialSSBOs[m_currentFrame]->LoadData(materialDatas.data(), materialDatas.size() * sizeof(MaterialData));
    }

    void Renderer::UpdateOnFramebufferResized()
    {
        m_swapchain->Recreate();

        for (size_t i = 0; i < m_gBuffers.size(); i++)
        {
            m_gBuffers[i]->Recreate(*m_device, m_swapchain->GetExtent(), m_descriptorPool);
        }
    }

	void Renderer::CreateInstance() 
    {
        // Optional appInfo struct
        const vk::ApplicationInfo appInfo
        {
            .pApplicationName = m_app.GetName().c_str(),
            .applicationVersion = VK_MAKE_VERSION(1,0,0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1,0,0),
            .apiVersion = vk::ApiVersion14
        };

        // List available instance layers and extensions
        auto layerProperties = m_context.enumerateInstanceLayerProperties();
        auto extensionProperties = m_context.enumerateInstanceExtensionProperties();
       
        /*
        #ifdef _DEBUG
            std::cout << "Available instance layer:" << std::endl;
            for (const auto& layer : layerProperties) 
                std::cout << '\t' << layer.layerName << std::endl;
            std::cout << "Available instance extensions:" << std::endl;
            for (const auto& extension : extensionProperties)
                std::cout << '\t' << extension.extensionName << std::endl;
        #endif
        */

        // Get the required instance LAYERS
        std::vector<char const*> requiredLayers;
        // Validation layers (optional)
        if (enableValidationLayers)
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        
        // NOTE: add here additional LAYERS if needed (append to requiredLayers)
        
        // Check if the required LAYERS are supported by the Vulkan implementation
        if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
            return std::ranges::none_of(layerProperties,
                [requiredLayer](auto const& layerProperty)
                { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
            }))
        {
            throw std::runtime_error("One or more required layers are not supported!");
        }

        // Get the required instance EXTENSIONS from GLFW (e.g. VK_KHR_surface, VK_KHR_win32_surface)
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        // Check if the required GLFW extensions are supported by the Vulkan implementation
        for (uint32_t i = 0; i < glfwExtensionCount; i++) {
            if (std::ranges::none_of(extensionProperties,
                [glfwExtension = glfwExtensions[i]](auto const& extensionProperty)
                { return strcmp(extensionProperty.extensionName, glfwExtension) == 0; }))
            {
                throw std::runtime_error("Required GLFW extension not supported: " + std::string(glfwExtensions[i]));
            }
        }

        // NOTE: add here additional EXTENSIONS if needed

        // Vulkan instance creation
        vk::InstanceCreateInfo createInfo
        {
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames = requiredLayers.data(),
            .enabledExtensionCount = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions
        };

        try
        {
            m_instance = vk::raii::Instance(m_context, createInfo);
        }
        catch (const vk::SystemError& err)
        {
            std::cerr << "Vulkan error: " << err.what() << std::endl;
        }
    }

    void Renderer::CreateSurface()
    {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*m_instance, m_window.GetHandle(), nullptr, &_surface) != 0)
            throw std::runtime_error("Failed to create window surface!");
        m_surface = vk::raii::SurfaceKHR(m_instance, _surface);
    }

    void Renderer::CreateDevice()
    {
        m_device = std::make_unique<Device>(m_instance, m_surface);
    }

    void Renderer::CreateSwapchain()
    {
        m_swapchain = std::make_unique<Swapchain>(*m_device, m_window, m_surface);
    }

    void Renderer::CreateGBuffer()
    {
        for (size_t i = 0; i < m_gBuffers.size(); i++)
        {
            m_gBuffers[i] = std::make_unique<GBuffer>(
                *m_device,
                m_swapchain->GetExtent(),
                m_descriptorPool
            );
        }
    }

    void Renderer::CreateDescriptorSetLayouts()
    {
        // Camera set layout
        // Binding 0 -> CameraData
        vk::DescriptorSetLayoutBinding cameraBinding {
            .binding = 0, 
            .descriptorType = vk::DescriptorType::eUniformBuffer, 
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 
            .pImmutableSamplers = nullptr
        };
        vk::DescriptorSetLayoutCreateInfo cameraLayout {
            .bindingCount = 1,
            .pBindings = &cameraBinding
        };
        m_cameraSetLayout = vk::raii::DescriptorSetLayout(m_device->GetDevice(), cameraLayout);

        // Lights set layout
        // Binding 0 -> LightsUBO
        vk::DescriptorSetLayoutBinding lightBinding {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };
        vk::DescriptorSetLayoutCreateInfo lightLayout {
            .bindingCount = 1,
            .pBindings = &lightBinding
        };
        m_lightSetLayout = vk::raii::DescriptorSetLayout(m_device->GetDevice(), lightLayout);

        // Object set layout
        // Binding 0 -> ObjectData
        vk::DescriptorSetLayoutBinding objectBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .pImmutableSamplers = nullptr
        };
        vk::DescriptorSetLayoutCreateInfo objectLayout{
            .bindingCount = 1,
            .pBindings = &objectBinding
        };
        m_objectSetLayout = vk::raii::DescriptorSetLayout(m_device->GetDevice(), objectLayout);

        // Material set layout
        // Binding 0 -> MaterialData
        vk::DescriptorSetLayoutBinding materialBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };
        vk::DescriptorSetLayoutCreateInfo materialLayout{
            .bindingCount = 1,
            .pBindings = &materialBinding
        };
        m_materialSetLayout = vk::raii::DescriptorSetLayout(m_device->GetDevice(), materialLayout);

        // Textures and sampler set layouts
        // Binding 0 -> Samplers array
        // Binding 1 -> Textures array
        // Binding 2 -> Skybox
        constexpr uint32_t bindingCount = 3;
        std::array<vk::DescriptorSetLayoutBinding, bindingCount> bindings;
        bindings[0] = {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = MAX_SAMPLERS,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };
        bindings[1] = {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = MAX_TEXTURES,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };
        bindings[2] = {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };
        vk::DescriptorSetLayoutCreateInfo textureLayout
        {
            .bindingCount = bindingCount,
            .pBindings = bindings.data()
        };
        m_textureSetLayout = vk::raii::DescriptorSetLayout(m_device->GetDevice(), textureLayout);
    }

    void Renderer::CreatePushConstant()
    {
        m_objectPushConst.stageFlags = vk::ShaderStageFlagBits::eVertex;
        m_objectPushConst.offset = 0;
        m_objectPushConst.size = sizeof(ObjectPushConst);
    }

    void Renderer::CreatePipeline()
    {
        auto& gBuffer = m_gBuffers[0];

        // ---- GEOMETRY PASS ----
        PipelineBuilder pipelineBuilder{ *m_device };
        pipelineBuilder.EnableVertexInput();
        pipelineBuilder.EnableDepthTest();
        pipelineBuilder.DisableBackfaceCulling();
        std::vector<PipelineBuilder::ShaderStageInfo> geomShaderStages{
            {"./shaders/geometry_pass.vert.spv", vk::ShaderStageFlagBits::eVertex},
            {"./shaders/geometry_pass.frag.spv", vk::ShaderStageFlagBits::eFragment}
        };
        pipelineBuilder.SetShaderStages(geomShaderStages);
        pipelineBuilder.SetColorBlending(static_cast<uint32_t>(gBuffer->GetAttachmentsCount() - 1)); // Depth attachment doesn't need blending!

        std::vector<vk::DescriptorSetLayout> layouts{ m_cameraSetLayout, m_objectSetLayout, m_materialSetLayout, m_textureSetLayout };
        std::vector<vk::PushConstantRange> ranges{ m_objectPushConst };
        pipelineBuilder.SetPipelineLayout(layouts, ranges);
        pipelineBuilder.SetAttachmentsFormat(gBuffer->GetColorAttachmentFormats(), gBuffer->GetDepthFormat());

        auto [geomPipeline, geomPipelineLayout] = pipelineBuilder.BuildPipeline();
        m_defGeometryPipeline = std::move(geomPipeline);
        m_defGeometryPipelineLayout = std::move(geomPipelineLayout);

        // ---- LIGHTING PASS ----
        pipelineBuilder.Reset();
        std::vector<PipelineBuilder::ShaderStageInfo> lightShaderStages{
            {"./shaders/lighting_pass.vert.spv", vk::ShaderStageFlagBits::eVertex},
            {"./shaders/lighting_pass.frag.spv", vk::ShaderStageFlagBits::eFragment}
        };
        pipelineBuilder.SetShaderStages(lightShaderStages);
        pipelineBuilder.DisableVertexInput();
        pipelineBuilder.DisableDepthTest();
        pipelineBuilder.DisableBackfaceCulling(); // To avoid culling the fullscreen triangle
        pipelineBuilder.SetColorBlending(1); // 1 attachment -> swapchain image
        pipelineBuilder.SetPipelineLayout(
            std::vector<vk::DescriptorSetLayout>{ m_cameraSetLayout, m_lightSetLayout, gBuffer->GetDescriptorSetLayout(), m_textureSetLayout },
            std::vector<vk::PushConstantRange>{}
        );
        pipelineBuilder.SetAttachmentsFormat(std::vector<vk::Format>{ m_swapchain->GetSurfaceFormat().format }, vk::Format::eUndefined); // no depth attachment
        
        auto [lightPipeline, lightPipelineLayout] = pipelineBuilder.BuildPipeline();
        m_defLightingPipeline = std::move(lightPipeline);
        m_defLightingPipelineLayout = std::move(lightPipelineLayout);
    }

    void Renderer::CreateCommandPool()
    {
        // CommandPoolCreateInfo
        vk::CommandPoolCreateInfo poolInfo = {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_device->GetGraphicsQueueFamilyIndex()
        };
        m_commandPool = vk::raii::CommandPool(m_device->GetDevice(), poolInfo);
    }

    void Renderer::CreateCommandBuffer()
    {
        // CommandBufferAllocateInfo
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = m_commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };
        m_commandBuffers = vk::raii::CommandBuffers(m_device->GetDevice(), allocInfo);
    }

    // Create and store default samplers
    // - one sampler with repeated wrapping and no filtering
    // - one sampler with repeated wrapping and bilinear filtering
    //
    // Additional notes:
    // From glTF 2.0 specs: `When texture.sampler is undefined, 
    // a sampler with repeat wrapping (in both directions) 
    // and auto filtering MUST be used.` -> any of the above
    // can be used as a fallback/default sampler
    void Renderer::CreateSamplers()
    {
        std::array<vk::SamplerCreateInfo, MAX_SAMPLERS> createInfos{};
        createInfos[0] = {
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::False,
            .maxAnisotropy = 1.0f,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueBlack,
            .unnormalizedCoordinates = vk::False,
        };
        createInfos[1] = {
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::False,
            .maxAnisotropy = 1.0f,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueBlack,
            .unnormalizedCoordinates = vk::False,
        };

        for (size_t i = 0; i < m_samplers.size(); i++)
            m_samplers[i].emplace(m_device->GetDevice(), createInfos[i], nullptr);

        LOG("[Renderer] Initialized samplers!");
    }

    void Renderer::CreateUniformBuffers()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            // Camera uniform buffer
            vk::BufferCreateInfo cameraUboInfo{};
            cameraUboInfo.size = sizeof(CameraData);
            cameraUboInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
            VmaAllocationCreateInfo cameraUboAllocInfo{};
            cameraUboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            m_cameraUBOs[i] = std::make_unique<Buffer>(m_device->GetAllocator(), cameraUboInfo, cameraUboAllocInfo, true);

            // Lights uniform buffer
            vk::BufferCreateInfo lightsUboInfo{};
            lightsUboInfo.size = sizeof(LightsUBO);
            lightsUboInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
            VmaAllocationCreateInfo lightsUboAllocInfo{};
            lightsUboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            m_lightUBOs[i] = std::make_unique<Buffer>(m_device->GetAllocator(), lightsUboInfo, lightsUboAllocInfo, true);

            // Object data storage buffer creation
            vk::BufferCreateInfo objectSsboInfo{};
            objectSsboInfo.size = sizeof(ObjectData) * MAX_OBJECTS;
            objectSsboInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
            VmaAllocationCreateInfo objectSsboAllocInfo{};
            objectSsboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            m_objectSSBOs[i] = std::make_unique<Buffer>(m_device->GetAllocator(), objectSsboInfo, objectSsboAllocInfo, true);

            // Material data storage buffer creation
            vk::BufferCreateInfo materialSsboInfo{};
            materialSsboInfo.size = sizeof(MaterialData) * MAX_MATERIALS;
            materialSsboInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
            VmaAllocationCreateInfo materialSsboAllocInfo{};
            materialSsboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            m_materialSSBOs[i] = std::make_unique<Buffer>(m_device->GetAllocator(), materialSsboInfo, materialSsboAllocInfo, true);
        }
    }

    void Renderer::CreateDescriptorPool()
    {
        // TODO: remove hardcoded number of attachments
        // NOTE: the pool is created before the GBuffer because it
        // uses the pool to allocate the attachments sets
        uint32_t attachmentsCount = 3 * MAX_FRAMES_IN_FLIGHT;
        std::array<vk::DescriptorPoolSize, MAX_DESCRIPTOR_SETS> poolSizes {
            // Camera
            vk::DescriptorPoolSize { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            
            // Lights
            vk::DescriptorPoolSize { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },

            // Objects and materials
            vk::DescriptorPoolSize { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 2 * MAX_FRAMES_IN_FLIGHT },
            
            // GBuffer
            vk::DescriptorPoolSize { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = attachmentsCount },

            // Texture
            // These reserved space will be used by ONE descriptor set (see below)
            vk::DescriptorPoolSize { .type = vk::DescriptorType::eSampledImage, .descriptorCount = MAX_TEXTURES + 1 },
            vk::DescriptorPoolSize { .type = vk::DescriptorType::eSampler, .descriptorCount = MAX_SAMPLERS }
        };
        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * MAX_DESCRIPTOR_SETS,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        m_descriptorPool = vk::raii::DescriptorPool(m_device->GetDevice(), poolInfo);
    }

    // Descriptor sets allocation from the main pool
    void Renderer::AllocateDescriptorSets()
    {
        // Camera
        std::vector<vk::DescriptorSetLayout> cameraLayouts(MAX_FRAMES_IN_FLIGHT, m_cameraSetLayout);
        vk::DescriptorSetAllocateInfo cameraAllocInfo{
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(cameraLayouts.size()),
            .pSetLayouts = cameraLayouts.data()
        };
        m_cameraDescriptorSets = m_device->GetDevice().allocateDescriptorSets(cameraAllocInfo);

        // Lights
        std::vector<vk::DescriptorSetLayout> lightLayouts(MAX_FRAMES_IN_FLIGHT, m_lightSetLayout);
        vk::DescriptorSetAllocateInfo lightAllocInfo{
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(lightLayouts.size()),
            .pSetLayouts = lightLayouts.data()
        };
        m_lightDescriptorSets = m_device->GetDevice().allocateDescriptorSets(lightAllocInfo);

        // Object
        std::vector<vk::DescriptorSetLayout> objectLayouts(MAX_FRAMES_IN_FLIGHT, m_objectSetLayout);
        vk::DescriptorSetAllocateInfo objectAllocInfo{
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(objectLayouts.size()),
            .pSetLayouts = objectLayouts.data()
        };
        m_objectDescriptorSets = m_device->GetDevice().allocateDescriptorSets(objectAllocInfo);

        // Material
        std::vector<vk::DescriptorSetLayout> materialLayouts(MAX_FRAMES_IN_FLIGHT, m_materialSetLayout);
        vk::DescriptorSetAllocateInfo materialAllocInfo{
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(materialLayouts.size()),
            .pSetLayouts = materialLayouts.data()
        };
        m_materialDescriptorSets = m_device->GetDevice().allocateDescriptorSets(materialAllocInfo);

        // Texture and sampler
        vk::DescriptorSetAllocateInfo textureAllocInfo
        {
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &*m_textureSetLayout
        };
        m_textureDescriptorSets = std::move(m_device->GetDevice().allocateDescriptorSets(textureAllocInfo).front());
    }

    void Renderer::CreateSyncObjects()
    {
        m_imageAvailableSemaphores.clear();
        m_renderFinishedSemaphores.clear();
        m_inFlightFences.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            m_imageAvailableSemaphores.emplace_back(m_device->GetDevice(), vk::SemaphoreCreateInfo());
            m_renderFinishedSemaphores.emplace_back(m_device->GetDevice(), vk::SemaphoreCreateInfo());
            m_inFlightFences.emplace_back(m_device->GetDevice(), vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void Renderer::DrawObject(const Object& obj, uint32_t& idx)
    {
        // TODO: improve invalid ResourceIDs handling

        // Iterate through all the meshes associated with obj
        // NOTE: this loop will be skipped if the object has no mesh
        for (MeshID meshId : obj.GetMeshes())
        {
            const Mesh& mesh = ResourceManager::GetInstance().GetMesh(meshId);

            // NOTE:
            // .objectIndex is per-object
            // .materialIndex is per-mesh
            ObjectPushConst pc{
                .objectIndex = idx,
                .materialIndex = m_materialIDToSSBOID[m_currentFrame][mesh.GetMaterial()]
            };
            m_commandBuffers[m_currentFrame].pushConstants(
                *m_defGeometryPipelineLayout,
                vk::ShaderStageFlagBits::eVertex,
                0,
                vk::ArrayProxy<const ObjectPushConst>(1, &pc)
            );

            // Bind vertex and index buffer
            m_commandBuffers[m_currentFrame].bindVertexBuffers(0, mesh.GetVertexBuffer().GetHandle(), { 0 });
            m_commandBuffers[m_currentFrame].bindIndexBuffer(mesh.GetIndexBuffer().GetHandle(), 0, mesh.GetIndexType());

            // Draw call
            m_commandBuffers[m_currentFrame].drawIndexed(mesh.GetIndexBufferSize(), 1, 0, 0, 0);
        }

        // Increment index AFTER drawing the object
        if (obj.GetMeshes().size())
            ++idx;

        // Iterate through its children
        for (const auto& childPtr : obj.GetChildren())
        {
            const Object& child = *childPtr;
            DrawObject(child, idx);
        }
    }

    void Renderer::RecordCommandBuffer(uint32_t imageIndex)
    {
        auto& cmdBuf = m_commandBuffers[m_currentFrame];
        auto& gBuffer = m_gBuffers[m_currentFrame];
        vk::Extent2D swapchainExtent = m_swapchain->GetExtent();

        cmdBuf.begin({});
        
        // ---- Geometry pass ----

        // Transition G-buffer attachments to color attachment layout
        const GBuffer::Attachment* depthAttachment = nullptr;
        for (auto& attachment : gBuffer->GetAttachments())
        {
            if (attachment.type == GBuffer::AttachmentType::Depth)
            {
                depthAttachment = &attachment;
                continue;
            }
                
            TransitionImageLayout(
                attachment.image->GetHandle(), attachment.image->GetFormat(),
                vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                {}, vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput
            );
        }

        if(depthAttachment == nullptr)
            throw std::runtime_error("[RENDERER] Couldn't retrieve the G-buffer depth attachment because there's no such attachment!");

        // Transition depth attachment as well
        TransitionImageLayout(
            depthAttachment->image->GetHandle(), depthAttachment->image->GetFormat(),
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests
        );

        // Setup rendering info (dynamic rendering)
        // Color attachments rendering info
        std::vector<vk::RenderingAttachmentInfo> colorAttachmentInfos;
        for (auto& attachment : gBuffer->GetAttachments())
        {
            if (attachment.type == GBuffer::AttachmentType::Depth)
                continue;
            colorAttachmentInfos.push_back({
                .imageView = attachment.image->GetImageView(),
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
            });
        }
        // Depth attachment rendering info
        vk::RenderingAttachmentInfo depthAttachmentInfo{
            .imageView = depthAttachment->image->GetImageView(),
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearDepthStencilValue{1.0f, 0} // {depth, stencil} -> 1.0f - far plane
        };
        vk::RenderingInfo renderingInfo = {
            .renderArea = {.offset = { 0, 0 }, .extent = swapchainExtent},
            .layerCount = 1,
            .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentInfos.size()),
            .pColorAttachments = colorAttachmentInfos.data(),
            .pDepthAttachment = &depthAttachmentInfo
        };

        // Begin rendering
        cmdBuf.beginRendering(renderingInfo);
        
        // Bind the graphic pipeline (the attachment will be bound to the fragment shader output)
        m_commandBuffers[m_currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, m_defGeometryPipeline);

        // Set viewport and scissor size (dynamic rendering)
        m_commandBuffers[m_currentFrame].setViewport(
            0,
            vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f)
        );
        m_commandBuffers[m_currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));

        // Bind descriptor sets (camera UBO, object SSBO, texture and sampler arrays)
        m_commandBuffers[m_currentFrame].bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, m_defGeometryPipelineLayout, 0,
            {   
                m_cameraDescriptorSets[m_currentFrame], 
                m_objectDescriptorSets[m_currentFrame], 
                m_materialDescriptorSets[m_currentFrame], 
                m_textureDescriptorSets // shared between frames in flight
            },
            nullptr
        );

        // Draw all the objects
        // Referenced index used to point each object
        // to the correct data it needs (depth-first traversal)
        uint32_t idx = 0;
        for (const auto& objPtr : m_scene.GetObjects())
        {
            const Object& obj = *objPtr;
            DrawObject(obj, idx);
        }

        cmdBuf.endRendering();
    
        // ---- Lighting pass ----
        
        // Transition G-buffer attachments to shader read layout for sampling
        for (auto& attachment : gBuffer->GetAttachments())
        {
            TransitionImageLayout(
                attachment.image->GetHandle(),
                attachment.image->GetFormat(),
                attachment.type == GBuffer::AttachmentType::Depth ? vk::ImageLayout::eDepthAttachmentOptimal : vk::ImageLayout::eColorAttachmentOptimal, 
                vk::ImageLayout::eShaderReadOnlyOptimal,
                attachment.type == GBuffer::AttachmentType::Depth ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite : vk::AccessFlagBits2::eColorAttachmentWrite,
                attachment.type == GBuffer::AttachmentType::Depth ? vk::AccessFlagBits2::eDepthStencilAttachmentRead : vk::AccessFlagBits2::eColorAttachmentRead,
                attachment.type == GBuffer::AttachmentType::Depth ? vk::PipelineStageFlagBits2::eEarlyFragmentTests : vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eFragmentShader
            );
        }

        // Transition swapchain image to color attachment layout
        TransitionImageLayout(
            m_swapchain->GetImages()[imageIndex],
            m_swapchain->GetSurfaceFormat().format,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        // Setup rendering info
        vk::RenderingAttachmentInfo finalAttachmentInfo{
            .imageView = m_swapchain->GetImageViews()[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
        };
        vk::RenderingInfo lightingRenderingInfo = {
            .renderArea = {.offset = { 0, 0 }, .extent = swapchainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &finalAttachmentInfo
        };

        // Rendering (computing lighting)
        cmdBuf.beginRendering(lightingRenderingInfo);

        // Bind the graphic pipeline (the attachment will be bound to the fragment shader output)
        m_commandBuffers[m_currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, m_defLightingPipeline);

        // Set viewport and scissor size (dynamic rendering)
        m_commandBuffers[m_currentFrame].setViewport(
            0,
            vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f)
        );
        m_commandBuffers[m_currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));

        // Bind descriptor sets (camera UBO, G-buffer)
        m_commandBuffers[m_currentFrame].bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, m_defLightingPipelineLayout, 0,
            { m_cameraDescriptorSets[m_currentFrame], m_lightDescriptorSets[m_currentFrame], gBuffer->GetDescriptorSet(), m_textureDescriptorSets }, 
            nullptr
        );

        // Draw a triangle that covers the screen (optimization of a quad)
        cmdBuf.draw(3, 1, 0, 0);

        // Draw Dear ImGui
        DrawImGuiFrame(ImGui::GetDrawData());

        cmdBuf.endRendering();

        // After rendering, transition the swapchain image to PRESENT_SRC
        TransitionImageLayout(
            m_swapchain->GetImages()[imageIndex],
            m_swapchain->GetSurfaceFormat().format,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        cmdBuf.end();
    }

    void Renderer::TransitionImageLayout(
        vk::Image image,
        vk::Format imageFormat,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags2 srcAccessMask,
        vk::AccessFlags2 dstAccessMask,
        vk::PipelineStageFlags2 srcStageMask,
        vk::PipelineStageFlags2 dstStageMask
    )
    {
        // Infer aspect from the image format
        // TODO: implement appropriate function
        vk::ImageAspectFlags aspect{};
        switch (imageFormat)
        {
        case vk::Format::eD16Unorm:
        case vk::Format::eD32Sfloat:
            aspect = vk::ImageAspectFlagBits::eDepth;
            break;
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            aspect = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            break;
        default:
            aspect = vk::ImageAspectFlagBits::eColor;
            break;
        }

        // Create and submit barrier
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        m_commandBuffers[m_currentFrame].pipelineBarrier2(dependencyInfo);
    }
}