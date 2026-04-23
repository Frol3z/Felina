#pragma once

#include "Common.hpp"

#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <vector>

namespace Felina 
{
	class Buffer;
	class Device;

	struct Vertex
	{
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;

		static vk::VertexInputBindingDescription GetBindingDescription()
		{
			return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
		}

		static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescriptions()
		{
			return
			{
				vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
				vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)),
				vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv))
			};
		}
	};

	class Mesh
	{
		public:
			// TODO: replace the optional material with a default material
			Mesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, MaterialID material = (MaterialID) -1);
			~Mesh();

			void Load(Device& device);
			void Unload();
			
			inline const Buffer& GetVertexBuffer() const { 
				assert(m_vertexBuffer && "Vertex buffer not initialized");
				return *m_vertexBuffer;
			};
			inline const Buffer& GetIndexBuffer() const { 
				assert(m_indexBuffer && "Index buffer not initialized");
				return *m_indexBuffer;
			};
			inline size_t GetIndexCount() const { return m_indices.size(); };
			inline vk::IndexType GetIndexType() const { return vk::IndexType::eUint32; };
			inline size_t GetVertexCount() const { return m_vertices.size(); };

			inline void SetMaterial(MaterialID newMaterial) { m_material = newMaterial; };
			inline MaterialID GetMaterial() const { return m_material; };

		private:
			void CreateStagingBuffer(Device& device, vk::DeviceSize size);
			void DestroyStagingBuffer();
			void CreateVertexBuffer(Device& device, vk::DeviceSize size);
			void CreateIndexBuffer(Device& device, vk::DeviceSize size);

			std::vector<Vertex> m_vertices;
			std::vector<uint32_t> m_indices;

			std::unique_ptr<Buffer> m_stagingBuffer = nullptr;
			std::unique_ptr<Buffer> m_vertexBuffer = nullptr;
			std::unique_ptr<Buffer> m_indexBuffer = nullptr;

			MaterialID m_material = (MaterialID) -1;
	};
}