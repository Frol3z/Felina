#pragma once

#include "Camera.hpp"
#include "Mesh.hpp"
#include "Object.hpp"

#include <unordered_map>
#include <memory>
#include <string>

namespace Felina 
{
	class Scene
	{
		public:
			Scene(float viewportWidth, float viewportHeight);
			~Scene();

			template <typename Fn>
			static void Traverse(const Object& obj, const glm::mat4& parent, Fn&& fn)
			{
				glm::mat4 model = parent * obj.GetModelMatrix();

				fn(obj, model);

				for (const auto& child : obj.GetChildren())
					Traverse(*child, model, fn);
			}

			Camera& GetCamera() { return m_camera; }
			const Camera& GetCamera() const { return m_camera; }

			// TODO: update
			void AddObject(std::unique_ptr<Object> object);
			inline const std::vector<std::unique_ptr<Object>>& GetObjects() const { return m_objects; }
			inline void ClearObjects() { m_objects.clear(); }

		private:
			Camera m_camera;
			std::vector<std::unique_ptr<Object>> m_objects; // Top-level objects
	};
}