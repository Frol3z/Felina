#pragma once

#include <glm/glm.hpp>

namespace Felina
{
	// This definitions may be moved somewhere else
	// Coordinate system is right-handed Y-up
	constexpr glm::vec3 WORLD_UP{ 0.0, 1.0, 0.0 };
	constexpr glm::vec3 WORLD_RIGHT{ 1.0, 0.0, 0.0 };
	constexpr glm::vec3 WORLD_FORWARD{ 0.0, 0.0, -1.0 };
	constexpr glm::vec3 WORLD_ORIGIN{ 0.0, 0.0, 0.0 };

	class Camera
	{
		public:
			Camera(float width, float height, float nearPlane = 0.1f, float farPlane = 1000.0f);
		
			void Rotate(double azimuth, double elevation);
			void Pan(double deltaX, double deltaY);
			void Dolly(double amount);
			void UpdateProjectionMatrix(float newWidth, float newHeight);
			void SetPosition(glm::vec3 position);

			inline glm::vec3 GetPosition() const { return m_position; }
			inline glm::vec3 GetTarget() const { return m_target; }

			glm::mat4 GetProjectionMatrix() const;
			glm::mat4 GetViewMatrix() const;
			glm::mat4 GetInvViewProj() const;

			inline void SetExposure(float exposure) { m_exposure = exposure; }
			inline const float GetExposure() const { return m_exposure; }

			inline void SetSkyboxIntensity(float intensity) { m_skyboxIntensity = intensity; }
			inline const float GetSkyboxIntensity() const { return m_skyboxIntensity; }
	
		private:
			void ComputeLocalCoordinateSystem();
			void ComputeProjectionMatrix();
			void ComputeViewMatrix();
			void ComputeInvViewProj();

			glm::mat4 m_projectionMatrix;
			glm::mat4 m_viewMatrix;
			glm::mat4 m_invViewProj;

			// Camera/view space
			glm::vec3 m_position;
			glm::vec3 m_localUp;
			glm::vec3 m_localRight;
			glm::vec3 m_localForward;
			
			// Target position (world origin by default)
			glm::vec3 m_target;

			// Exposures
			float m_exposure = 0.005f; // scene exposure
			float m_skyboxIntensity = 0.2f;

			float m_left = 0.0f;
			float m_right = 0.0f;
			float m_bottom = 0.0f;
			float m_top = 0.0f;
			float m_near = 0.0f;
			float m_far = 0.0f;
	};
}