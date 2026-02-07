#include "Mesh.hpp"

#include "Device.hpp"
#include "Buffer.hpp"

namespace Felina
{
    Mesh::Mesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, MaterialID material)
        : m_vertices(vertices), m_indices(indices), m_material(material)
    {
    }

    Mesh::~Mesh()
    {
        Unload();
    }

	void Mesh::Load(Device& device)
	{
        auto vertexDataSize = m_vertices.size() * sizeof(Vertex);
        if (!m_indices.empty())
        {
            // The mesh supports indices
            auto indexDataSize = m_indices.size() * sizeof(m_indices[0]);

            // stagingSize should be equal to vertexDataSize most of the time
            auto stagingSize = std::max(vertexDataSize, indexDataSize);
            CreateStagingBuffer(device, stagingSize);
            CreateVertexBuffer(device, vertexDataSize);
            CreateIndexBuffer(device, indexDataSize);
        }
        else
        {
            // The mesh does not support indices -> vertices only representation
            CreateStagingBuffer(device, vertexDataSize);
            CreateVertexBuffer(device, vertexDataSize);
        }

        // Once this line is reached the data has been transferred
        // correctly to GPU memory, so the staging buffer can be safely destroyed
        // since it won't be needed anymore
        DestroyStagingBuffer();
	}

    void Mesh::Unload()
    {
        m_vertexBuffer.reset();
        m_indexBuffer.reset();
    }

    void Mesh::CreateStagingBuffer(Device& device, vk::DeviceSize size)
    {        
        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        m_stagingBuffer = std::make_unique<Buffer>(device.GetAllocator(), stagingInfo, stagingAllocInfo);
    }

    void Mesh::DestroyStagingBuffer()
    {
        m_stagingBuffer.reset();
    }

    void Mesh::CreateVertexBuffer(Device& device, vk::DeviceSize size)
    {
        VkBufferCreateInfo vertexInfo{};
        vertexInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vertexInfo.size = size;
        vertexInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vertexInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo vertexAllocInfo{};
        vertexAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        m_vertexBuffer = std::make_unique<Buffer>(device.GetAllocator(), vertexInfo, vertexAllocInfo);

        // Load vertex data on the staging buffer
        m_stagingBuffer->LoadData(m_vertices.data(), size);

        // Issue request to copy vertex data loaded on the staging buffer to GPU memory
        device.CopyBuffer(*m_stagingBuffer, *m_vertexBuffer, size);
    }

    void Mesh::CreateIndexBuffer(Device& device, vk::DeviceSize size)
    {
        VkBufferCreateInfo indexInfo{};
        indexInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        indexInfo.size = size;
        indexInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        indexInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo indexAllocInfo{};
        indexAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        m_indexBuffer = std::make_unique<Buffer>(device.GetAllocator(), indexInfo, indexAllocInfo);

        // Load index data on the staging buffer
        m_stagingBuffer->LoadData(m_indices.data(), size);

        // Issue request to copy index data loaded on the staging buffer to GPU memory
        device.CopyBuffer(*m_stagingBuffer, *m_indexBuffer, size);
    }
}