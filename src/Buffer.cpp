#include "Buffer.hpp"

namespace Felina
{
	Buffer::Buffer(const VmaAllocator& allocator, const VkBufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocInfo, bool isPersistent)
		: m_allocator(allocator), m_isPersistent(isPersistent)
	{
		VkBuffer raw;
		vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &raw, &m_allocation, nullptr);
		m_buffer = vk::Buffer(raw);

		if (m_isPersistent)
		{
			// If persistent mapping is enable, the buffer will hold a pointer to the mapped
			// memory block for its entire lifespan without the overhead of remapping each time
			// NOTE: use it with buffer which will be updated each frame
			vmaMapMemory(m_allocator, m_allocation, &m_persistentMappedMemory);
		}
	}

	Buffer::~Buffer()
	{
		if (m_buffer && m_allocation)
		{
			if (m_isPersistent)
				vmaUnmapMemory(m_allocator, m_allocation);

			vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
			m_buffer = nullptr;
			m_allocation = VK_NULL_HANDLE;
		}
	}
	
	void Buffer::LoadData(const void* data, const size_t size)
	{
		// TODO: Assert if we should be able to load data onto this buffer
		if (m_isPersistent)
		{
			memcpy(m_persistentMappedMemory, data, size);
		}
		else
		{
			void* mappedMemory = nullptr;
			vmaMapMemory(m_allocator, m_allocation, &mappedMemory);
			memcpy(mappedMemory, data, size);
			vmaUnmapMemory(m_allocator, m_allocation);
		}
	}

	vk::DeviceAddress Buffer::GetDeviceAddress(Device& device) const
	{
		vk::BufferDeviceAddressInfo addressInfo { .buffer = m_buffer };
		return device.GetDevice().getBufferAddress(addressInfo);
	}
}