#pragma once

#include "Mesh.hpp"
#include "Material.hpp"
#include "Texture.hpp"
#include "Common.hpp"

#include <unordered_map>

namespace Felina
{
	class Renderer;

	class ResourceManager
	{
		public:
			template<typename T>
			struct Resource {
				std::string name;
				std::unique_ptr<T> resource;
			};

		public:
			ResourceManager(const ResourceManager&) = delete; // Delete copy constructor
			ResourceManager& operator=(const ResourceManager&) = delete; // Delete assignment operator
			
			static ResourceManager& GetInstance()
			{
				static ResourceManager instance;
				return instance;
			}

			inline void SetDefaultMaterial(MaterialID mat) { m_defaultMaterial = mat; };
			inline MaterialID GetDefaultMaterial() const { return m_defaultMaterial; };

			MeshID LoadMesh(std::unique_ptr<Mesh> mesh, const std::string& name, Renderer& renderer);
			MaterialID LoadMaterial(std::unique_ptr<Material> material, const std::string& name);
			TextureID LoadTexture(std::unique_ptr<Texture> texture, const std::string& name,
				const void* rawImageData, size_t rawImageSize,
				Renderer& renderer
			);
			void UnloadAll();

			const Mesh& GetMesh(MeshID id) const;
			const std::string& GetMeshName(MeshID id) const;
			const std::unordered_map<MeshID, Resource<Mesh>>& GetMeshes() const { return m_meshes; }
			const Material& GetMaterial(MaterialID id) const;
			const std::string& GetMaterialName(MaterialID id) const;
			const std::unordered_map<MaterialID, Resource<Material>>& GetMaterials() const { return m_materials; }
			const std::unordered_map<TextureID, Resource<Texture>>& GetTextures() const { return m_textures; }
		private:
			ResourceManager(){} // Private constructor

			// Resources
			std::unordered_map<MeshID, Resource<Mesh>> m_meshes;
			std::unordered_map<MaterialID, Resource<Material>> m_materials;
			std::unordered_map<TextureID, Resource<Texture>> m_textures;
			MaterialID m_defaultMaterial{ 0 };

			// Id counters
			MeshID m_meshID{ 0 };
			MaterialID m_materialID{ 0 };
			TextureID m_textureID{ 0 };
	};
}