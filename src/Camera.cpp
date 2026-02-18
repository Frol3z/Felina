#include "Camera.hpp"

#include "Common.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Felina
{
	Camera::Camera(float width, float height, float nearPlane, float farPlane)
		: m_right(width), m_bottom(height), m_near(nearPlane), m_far(farPlane),
		m_projectionMatrix(1.0f), m_viewMatrix(1.0f),
		m_position(WORLD_ORIGIN), m_localUp(WORLD_UP), m_target(WORLD_ORIGIN)
	{
		ComputeLocalCoordinateSystem();
		ComputeViewMatrix();
		ComputeProjectionMatrix();
	}

	void Camera::Rotate(double azimuth, double elevation)
	{
		// TODO: fix snapping

		// Apply rotations
		glm::vec3 offset = m_position - m_target;
		glm::quat qAzimuth = glm::angleAxis(glm::radians((float)azimuth), glm::vec3(0.0f, 0.0f, 1.0f));
		offset = qAzimuth * offset;

		glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), offset));
		glm::quat qElevation = glm::angleAxis(glm::radians((float)elevation), right);
		offset = qElevation * offset;

		m_position = m_target + offset;

		ComputeLocalCoordinateSystem();
		ComputeViewMatrix();
	}

	void Camera::Pan(double deltaX, double deltaY)
	{
		glm::vec3 offset = static_cast<float>(deltaX) * m_localRight
			+ static_cast<float>(-deltaY) * m_localUp; // Sign flipped!
		m_position += offset;
		m_target += offset;
		ComputeLocalCoordinateSystem();
		ComputeViewMatrix();
	}

	void Camera::Dolly(double amount)
	{
		glm::vec3 offset = static_cast<float>(amount) * m_localForward;
		m_position += offset;
		m_target += offset;
		ComputeViewMatrix();
	}

	void Camera::UpdateProjectionMatrix(float newWidth, float newHeight)
	{
		m_right = newWidth;
		m_bottom = newHeight;
		ComputeProjectionMatrix();
	}

	void Camera::SetPosition(glm::vec3 position)
	{
		m_position = position;
		ComputeLocalCoordinateSystem();
		ComputeViewMatrix();
	}

	glm::mat4 Camera::GetProjectionMatrix() const
	{
		return m_projectionMatrix;
	}

	glm::mat4 Camera::GetViewMatrix() const
	{
		// Camera position, center of projection, up axis (Z-axis)
		return m_viewMatrix;
	}

	glm::mat4 Camera::GetInvViewProj() const
	{
		return m_invViewProj;
	}

	void Camera::ComputeLocalCoordinateSystem()
	{
		m_localForward = glm::normalize(m_target - m_position);
		if (m_localForward == WORLD_ORIGIN) 
		{
			// It may happen that if the camera is positioned at the
			// world origin, then the resulting forward vector would be
			// the null vector.
			// If that's the case, the world +Y is assumed
			// as the forward direction.
			m_localForward = WORLD_FORWARD;
		}
		m_localRight = glm::normalize(glm::cross(m_localForward, WORLD_UP));
		m_localUp = glm::normalize(glm::cross(m_localRight, m_localForward));
	}

	void Camera::ComputeViewMatrix()
	{
		// Convert to our world CS in OpenGL CS
		// NOTE: extrinsic parameters (world space -> view space)
		m_viewMatrix = glm::lookAt(m_position, m_target, m_localUp);
		ComputeInvViewProj();
	}

	void Camera::ComputeProjectionMatrix()
	{
		// NOTE: intrinsic parameters (view space -> clip space)
		m_projectionMatrix = glm::perspective(glm::radians(45.0f), m_right / m_bottom, m_near, m_far);
		m_projectionMatrix[1][1] *= -1; // Flips Y-axis (OpenGL CS -> Vulkan CS)
		ComputeInvViewProj();
	}

	void Camera::ComputeInvViewProj()
	{
		// TODO: avoid recomputing the inverse twice when both view and projection change
		m_invViewProj = glm::inverse(m_projectionMatrix * m_viewMatrix);
	}
}