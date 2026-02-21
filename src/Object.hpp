#pragma once

#include "ResourceManager.hpp"
#include "Transform.hpp"
#include "Light.hpp"

#include <string>

namespace Felina
{
	class Object
	{
		public:
			Object(const std::string& name, Object* parent = nullptr) : m_name(name), m_parent(parent) {}
			Object(const std::string& name, const std::vector<MeshID>& meshes, Object* parent = nullptr)
				: m_name(name), m_meshes(meshes), m_parent(parent) {}

			inline const std::string& GetName() const { return m_name; }
			inline const std::vector<MeshID>& GetMeshes() const { return m_meshes; }

			// Children
			void AddChild(std::unique_ptr<Object> child);
			const std::vector<std::unique_ptr<Object>>& GetChildren() const { return m_children; }

			// Transform
			void Translate(const glm::vec3& translation) { m_transform.Translate(translation);}
			void Rotate(const float angle, const glm::vec3& axis) { m_transform.Rotate(angle, axis); }
			void Scale(const glm::vec3& scale) { m_transform.Scale(scale); }
			void SetPosition(const glm::vec3& position) { m_transform.SetPosition(position); }
			void SetRotation(const glm::quat& rotation) { m_transform.SetRotation(rotation); }
			void SetScale(const glm::vec3& scale) { m_transform.SetScale(scale); }
			void SetModelMatrix(const glm::mat4& matrix) { m_transform.SetMatrix(matrix); }
			const glm::vec3& GetPosition() const { return m_transform.GetPosition(); }
			const glm::quat& GetRotation() const { return m_transform.GetRotation(); }
			const glm::vec3& GetScale() const { return m_transform.GetScale(); }
			inline glm::mat4 GetModelMatrix() const { return m_transform.GetMatrix(); }

			// Light
			inline void SetLightData(Light data) { m_lightData = data; }
			inline std::optional<Light> GetLightData() const { return m_lightData; }

		private:
			std::string m_name;
			Transform m_transform;
			std::vector<MeshID> m_meshes;

			Object* m_parent;
			std::vector<std::unique_ptr<Object>> m_children;

			std::optional<Light> m_lightData;
	};
}