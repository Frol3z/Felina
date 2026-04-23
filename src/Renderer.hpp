#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <optional>
#include <filesystem>

// Required for MaterialID and MeshID definitions
#include "ResourceManager.hpp"
#include "Common.hpp"

struct ImGui_ImplVulkan_InitInfo;
struct ImDrawData;

namespace Felina
{
	// Fwd declaration
	class Application;
	class Device;
	class Swapchain;
	class GBuffer;
	class Buffer;
	class Window;
	class Scene;
	class Object;
	class Mesh;

	class Renderer
	{
		public:
			struct CameraData
			{
				glm::vec3 position;
				glm::mat4 view;
				glm::mat4 proj;
				glm::mat4 invViewProj;
				float skyboxIntensity;
				float exposure;
			};

			// NOTE: everything is 16 bytes aligned
			struct LightData // 80 byte
			{
				uint32_t	type;			// DIRECTIONAL = 0, POINT = 1, SPOT = 2
				glm::vec3	color;
				glm::vec3	position;       // used by POINT and SPOT
				glm::vec3	direction;      // used by DIRECTIONAL and SPOT 
				
				float		intensity;
				float		range;
				float		innerConeAngle;
				float		outerConeAngle;
			};

			struct LightsUBO
			{
				std::array<LightData, MAX_LIGHTS> lights;
				uint32_t lightsCount;
				uint32_t padding[3];
			};

			struct MaterialData
			{
				glm::vec3 baseColor;
				glm::vec4 materialInfo;

				uint32_t baseColorTex;
				uint32_t materialInfoTex;
			};

			struct ObjectData
			{
				glm::mat4 model;
				glm::mat3 normal;
			};

			struct ObjectPushConst
			{
				uint32_t objectIndex;
				uint32_t materialIndex;
			};

			struct EquirectToCubemapPushConst
			{
				glm::mat4 proj;
				glm::mat4 view;
			};

			struct BLAS
			{
				vk::raii::AccelerationStructureKHR handle;
				std::unique_ptr<Buffer> storageBuffer;
			};

			// Max number of descriptor sets PER FRAME
			// Current sets:
			// - equirect to cubemap combined image sampler
			// - camera UBO
			// - lights UBO
			// - objects SSBO
			// - materials SSBO
			// - GBuffer (see GBuffer class)
			// - texture and sampler arrays
			static constexpr uint32_t MAX_DESCRIPTOR_SETS = 7;
			static constexpr uint32_t MAX_SAMPLERS = 2;

		public:
			Renderer(Application& app, const Window& window, const Scene& scene);
			~Renderer();

			void DrawFrame();
			void WaitIdle();
			void LoadMesh(Mesh& mesh);
			void LoadTexture(const Texture& texture, const void* rawImageData, size_t rawImageSize);
			void LoadSkybox(const std::filesystem::path& folderPath);
			inline const TextureID GetSkyboxCubemapId() const { return m_skyboxCubemap; }
			void UpdateDescriptorSets(); 

			const Device& GetDevice() const;

			// Dear ImGui
			ImGui_ImplVulkan_InitInfo GetImGuiInitInfo();
			void DrawImGuiFrame(ImDrawData* drawData);

			void BuildAccelerationStructure(); // move down (private)
		private:
			void SetupFrameData();
			void UpdateOnFramebufferResized();

			void CreateInstance();
			void CreateSurface();
			void CreateDevice();
			void CreateSwapchain();
			void CreateGBuffer();
			void CreateDescriptorSetLayouts();
			void CreatePushConstant();

			// Skybox & cubemap
			void CreateCubemap();
			void CreateEquirectToCubemapPipeline();
			void UpdateEquirectToCubemapDescriptorSet(const Texture& texture, const vk::Sampler sampler);
			void RecordEquirectToCubemapDescriptorSetCommandBuffer(const vk::raii::CommandBuffer& cmdBuf);
			
			void CreatePipeline();
			void CreateCommandPool();
			void CreateCommandBuffer();
			void CreateSamplers();
			void CreateUniformBuffers();
			void CreateDescriptorPool();
			void AllocateDescriptorSets();
			void CreateSyncObjects();
			
			// Ray-tracing and AS

			void DrawObject(const Object& obj, uint32_t& idx);
			void RecordCommandBuffer(uint32_t imageIndex); // 2 passes
			void TransitionImageLayout(
				vk::Image image,
				vk::Format imageFormat,
				vk::ImageLayout oldLayout,
				vk::ImageLayout newLayout,
				vk::AccessFlags2 srcAccessMask,
				vk::AccessFlags2 dstAccessMask,
				vk::PipelineStageFlags2 srcStageMask,
				vk::PipelineStageFlags2 dstStageMask
			);

			// NOTE: non-const reference because
			// Application::IsFramebufferResized cannot be const
			Application& m_app;
			const Window& m_window;
			const Scene& m_scene;

			uint32_t m_currentFrame = 0;
			// Look-up table to match the Material ID to the physical GPU storage buffer index
			std::array<std::unordered_map<MaterialID, uint32_t>, MAX_FRAMES_IN_FLIGHT> m_materialIDToSSBOID;
			// Look-up table to match the Texture ID to the GPU texture array index
			std::array<std::unordered_map<TextureID, uint32_t>, MAX_FRAMES_IN_FLIGHT> m_textureIDToArrayID;

			// Dear ImGui custom vertex shader (temporary fix for colors issue)
			std::vector<char> m_imGuiCustomVertShaderCode;
			vk::ShaderModuleCreateInfo m_imGuiCustomVertShaderModuleCreateInfo;

			vk::raii::Context m_context; // Used as a dynamic loader
			vk::raii::Instance m_instance = nullptr;
			vk::raii::SurfaceKHR m_surface = nullptr;
			std::unique_ptr<Device> m_device = nullptr;
			std::unique_ptr<Swapchain> m_swapchain = nullptr;
			vk::raii::DescriptorPool m_descriptorPool = nullptr;
			vk::raii::CommandPool m_commandPool = nullptr;
			std::array<std::unique_ptr<GBuffer>, MAX_FRAMES_IN_FLIGHT> m_gBuffers;

			// Equirect to cubemap pipeline stuff
			TextureID m_skyboxCubemap;
			vk::raii::DescriptorSetLayout m_equirectToCubemapSetLayout = nullptr;
			vk::PushConstantRange m_equirectToCubemapPushConst;
			vk::raii::PipelineLayout m_equirectToCubemapPipelineLayout = nullptr;
			vk::raii::Pipeline m_equirectToCubemapPipeline = nullptr;
			vk::raii::DescriptorSet m_equirectToCubemapDescriptorSet = nullptr;

			// Main deferred pipeline stuff
			vk::raii::DescriptorSetLayout m_cameraSetLayout = nullptr;
			vk::raii::DescriptorSetLayout m_lightSetLayout = nullptr;
			vk::raii::DescriptorSetLayout m_materialSetLayout = nullptr;
			vk::raii::DescriptorSetLayout m_objectSetLayout = nullptr;
			vk::raii::DescriptorSetLayout m_textureSetLayout = nullptr;
			vk::PushConstantRange m_objectPushConst;
			vk::raii::PipelineLayout m_defGeometryPipelineLayout = nullptr;
			vk::raii::Pipeline m_defGeometryPipeline = nullptr;
			vk::raii::PipelineLayout m_defLightingPipelineLayout = nullptr;
			vk::raii::Pipeline m_defLightingPipeline = nullptr;

			std::vector<vk::raii::CommandBuffer> m_commandBuffers;

			std::array<std::optional<vk::raii::Sampler>, MAX_SAMPLERS> m_samplers;

			std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_cameraUBOs;
			std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_lightUBOs;
			std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_objectSSBOs;
			std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_materialSSBOs;
			std::vector<vk::raii::DescriptorSet> m_cameraDescriptorSets;
			std::vector<vk::raii::DescriptorSet> m_lightDescriptorSets;
			std::vector<vk::raii::DescriptorSet> m_objectDescriptorSets;
			std::vector<vk::raii::DescriptorSet> m_materialDescriptorSets;
			// Just one shared between frames because it will be read-only
			vk::raii::DescriptorSet m_textureDescriptorSets = nullptr;

			// Acceleration structure
			std::vector<BLAS> m_blas;

			std::vector<vk::raii::Semaphore> m_imageAvailableSemaphores;
			std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
			std::vector<vk::raii::Fence> m_inFlightFences;
	};
}