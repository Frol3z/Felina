#include "Device.hpp"

#include "Buffer.hpp"
#include "Texture.hpp"

#include <iostream>
#include <set>

namespace Felina
{
	Device::Device(vk::raii::Instance& instance, const vk::raii::SurfaceKHR& surface)
	{
        SelectPhysicalDevice(instance, surface);
        CreateLogicalDevice(instance);
        CreateMemoryAllocator(instance);
        CreateImmediateCommandPool();
	}

    Device::~Device()
    {
        // VMA allocator cleanup
        if (m_allocator)
        {
            vmaDestroyAllocator(m_allocator);
            m_allocator = VK_NULL_HANDLE;
        }
    }

    void Device::CopyBuffer(const Buffer& srcBuffer, const Buffer& dstBuffer, vk::DeviceSize size)
    {
        // Short-lived command buffer allocation
        vk::CommandBufferAllocateInfo allocInfo
        {
            .commandPool = m_immediateCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        vk::raii::CommandBuffer cmdBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());

        // Record commands
        cmdBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        cmdBuffer.copyBuffer(srcBuffer.GetHandle(), dstBuffer.GetHandle(), vk::BufferCopy(0, 0, size));
        cmdBuffer.end();

        // Submit to queue and wait for completion
        m_graphicsQueue.submit(
            vk::SubmitInfo {
            .commandBufferCount = 1,
            .pCommandBuffers = &*cmdBuffer
            },
            nullptr
        );
        m_graphicsQueue.waitIdle();
    }

    void Device::CopyBufferToImage(const Buffer& src, const Texture& dst, vk::DeviceSize size)
    {
        // Short-lived command buffer allocation
        vk::CommandBufferAllocateInfo allocInfo
        {
            .commandPool = m_immediateCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        vk::raii::CommandBuffer cmdBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());

        // Record commands
        cmdBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        
        // Transition from undefined image layout to transfer dst layout using an ImageMemoryBarrier
        vk::ImageMemoryBarrier firstBarrier 
        {
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = dst.GetHandle(),
            .subresourceRange = dst.GetImageSubresourceRange()
        };
        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, nullptr, firstBarrier
        );

        // Memory transfer
        const auto range = dst.GetImageSubresourceRange();
        uint32_t layerCount = range.layerCount;
        uint32_t baseLayer = range.baseArrayLayer;
        vk::BufferImageCopy region
        {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                vk::ImageAspectFlagBits::eColor, // aspectMsk
                0, // mip
                baseLayer, // baseLayer
                layerCount // layerCount
        },
            .imageOffset = {0, 0, 0}, 
            .imageExtent = dst.GetExtent()
        };
        cmdBuffer.copyBufferToImage(src.GetHandle(), dst.GetHandle(), vk::ImageLayout::eTransferDstOptimal, region);
        
        // Transition from transfer dst layout to shader read optimal using another ImageMemoryBarrier
        vk::ImageMemoryBarrier secondBarrier
        {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = dst.GetHandle(),
            .subresourceRange = dst.GetImageSubresourceRange()
        };
        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, nullptr, secondBarrier
        );
        cmdBuffer.end();

        // Submit to queue and wait for completion
        m_graphicsQueue.submit(
            vk::SubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*cmdBuffer
            },
            nullptr
        );
        m_graphicsQueue.waitIdle();
    }

    vk::raii::CommandBuffer Device::CreateTransientCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = m_immediateCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        // NOTE: ownership will be moved to the caller
        return std::move(m_device.allocateCommandBuffers(allocInfo).front());
    }

    void Device::SelectPhysicalDevice(vk::raii::Instance& instance, const vk::raii::SurfaceKHR& surface)
    {
        for (const auto& physicalDevice : vk::raii::PhysicalDevices(instance))
        {
            auto deviceProperties = physicalDevice.getProperties();

            // Check for Vulkan 1.4+ support
            if (deviceProperties.apiVersion < VK_API_VERSION_1_4)
                continue;

            // Add other requirements check here if needed

            // Print infos if needed
            /*
            #ifdef _DEBUG
                std::cout << '\t' << "- " << deviceProperties.deviceName << std::endl;

                // Print physical device limits (TODO: remove if not needed anymore)
                std::cout <<
                    "maxBoundDescriptorSets: " <<
                    deviceProperties.limits.maxBoundDescriptorSets <<
                    std::endl <<
                    "maxUniformBufferRange: " <<
                    deviceProperties.limits.maxUniformBufferRange <<
                    std::endl <<
                    "maxStorageBufferRange: " <<
                    deviceProperties.limits.maxStorageBufferRange <<
                    std::endl;
            #endif
            */

            // Get required queue family
            auto queueFamilies = physicalDevice.getQueueFamilyProperties();
            int graphicsIndex = -1;
            int presentIndex = -1;
            for (size_t i = 0; i < queueFamilies.size(); i++)
            {
                auto& q = queueFamilies[i];
                
                // Graphics
                if ((q.queueFlags & vk::QueueFlagBits::eGraphics) && graphicsIndex == -1)
                    graphicsIndex = static_cast<int>(i);

                // Presentation
                if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface) && presentIndex == -1)
                    presentIndex = static_cast<int>(i);
            }

            if (graphicsIndex != -1 && presentIndex != -1)
            {
                m_physicalDevice = physicalDevice;
                m_graphicsQueueFamilyIndex = static_cast<uint32_t>(graphicsIndex);
                m_presentQueueFamilyIndex = static_cast<uint32_t>(presentIndex);

                return;
            }
        }

        throw std::runtime_error("No suitable Vulkan physical device with required queue families found!");
    }

    void Device::CreateLogicalDevice(vk::raii::Instance& instance)
    {
        // Queue(s) create info
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        // If the two queues are from the same family, it avoids creating redundant create info structs
        std::set<uint32_t> uniqueQueueFamilies = { m_graphicsQueueFamilyIndex, m_presentQueueFamilyIndex };
        float queuePriority = 0.0f;

        for (uint32_t queueFamilyIndex : uniqueQueueFamilies)
        {
            vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
                .queueFamilyIndex = queueFamilyIndex,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority
            };
            queueCreateInfos.push_back(deviceQueueCreateInfo);
        }

        // Create a chain of feature structures to enable multiple new FEATURES (on top of those of Vulkan 1.0) all at once
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
            vk::PhysicalDeviceRobustness2FeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        > // To be able to change dinamically some pipeline properties
            featureChain = {
                {},
                { .bufferDeviceAddress = true },
                {.synchronization2 = true, .dynamicRendering = true},
                {.extendedDynamicState = true},
                { .nullDescriptor = true },
                { .accelerationStructure = true }
        };

        // Required device EXTENSIONS
        // TODO: actually check for support and don't rely on the validation layer
        std::vector<const char*> deviceExtensions = {
            // Mandatory extension for presenting framebuffer on a window
            vk::KHRSwapchainExtensionName,

            // Vulkan tutorial extensions
            vk::KHRSpirv14ExtensionName,
            vk::KHRSynchronization2ExtensionName,
            vk::KHRCreateRenderpass2ExtensionName,
            vk::KHRShaderDrawParametersExtensionName, // Required to be able to use SV_VertexID in shader code

            // RT-related extensions
            vk::KHRAccelerationStructureExtensionName,
            vk::KHRDeferredHostOperationsExtensionName, // Required by the extension above
            //vk::KHRRayTracingPipelineExtensionName,
            //vk::KHRRayQueryExtensionName
        };

        // Device creation
        vk::DeviceCreateInfo deviceCreateInfo{
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
        };

        try
        {
            m_device = vk::raii::Device(m_physicalDevice, deviceCreateInfo);
        }
        catch (const vk::SystemError& err)
        {
            std::cerr << "Vulkan error: " << err.what() << std::endl;
        }

        // Get graphics and presentation queue references
        m_graphicsQueue = vk::raii::Queue(m_device, m_graphicsQueueFamilyIndex, 0);
        m_presentQueue = vk::raii::Queue(m_device, m_presentQueueFamilyIndex, 0);
    }

    void Device::CreateMemoryAllocator(vk::raii::Instance& instance)
    {
        VmaAllocatorCreateInfo allocatorCreateInfo
        {
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = *m_physicalDevice,
            .device = *m_device,
            .instance = *instance,
            .vulkanApiVersion = vk::ApiVersion14, // TODO: remove this hardcoded api version
        };
        vmaCreateAllocator(&allocatorCreateInfo, &m_allocator);
    }

    void Device::CreateImmediateCommandPool()
    {
        // Creating a dedicated command pool used for memory transfer ops
        vk::CommandPoolCreateInfo poolInfo {
            // Specify that cmd buffer allocated from this pool will be short-lived
            .flags = vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = m_graphicsQueueFamilyIndex
        };
        m_immediateCommandPool = vk::raii::CommandPool(m_device, poolInfo);
    }
}