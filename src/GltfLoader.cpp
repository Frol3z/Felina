#include "GltfLoader.hpp"

#include "tiny_gltf.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "Scene.hpp"
#include "Renderer.hpp"
#include "ResourceManager.hpp"
#include "Common.hpp"

namespace Felina
{
	// Load all meshes in `model` and fill `meshes` with the corresponding MeshIDs
	static void LoadMeshes(tinygltf::Model& model, Renderer& renderer, 
		std::unordered_map<int, MaterialID>& materials, 
		std::unordered_map<int, std::vector<MeshID>>& meshes
	)
	{
		auto& rm = ResourceManager::GetInstance();
		
		// Iterate through all meshes and load them
		size_t i = 0; // mesh glTF index
		for (auto& mesh : model.meshes)
		{
			std::vector<MeshID> primitives;
			primitives.reserve(mesh.primitives.size());

			// Iterate through primitives
			size_t j = 0;
			for (auto& primitive : mesh.primitives)
			{
				// NOTE: only mode currently supported is TRIANGLE_LIST (see PipelineBuilder.cpp)
				if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
					throw std::runtime_error("[GltfLoader] Unsupported mode required!");

				// Loading vertices
				std::vector<Vertex> vertices;

				{
					// POSITION and NORMAL are mandatory
					// TEXCOORD_0 is optional and if absent it's filled with dummy values
					
					// Vertex position
					auto posIt = primitive.attributes.find("POSITION");
					if(posIt == primitive.attributes.end())
						throw std::runtime_error("[GltfLoader] Mandatory POSITION attribute missing!");
					auto& posAccessor = model.accessors[posIt->second];
					auto& posBufferView = model.bufferViews[posAccessor.bufferView];
					auto& posBuffer = model.buffers[posBufferView.buffer];
					assert(posAccessor.type == TINYGLTF_TYPE_VEC3 && "[GltfLoader] Unexpected type found for vertex position!");
					assert(posAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && "[GltfLoader] Unexpected componentType found for vertex position!");

					const uint8_t* posStart = posBuffer.data.data() + posBufferView.byteOffset + posAccessor.byteOffset;
					size_t posStride = posAccessor.ByteStride(posBufferView);

					// Vertex normal
					auto normIt = primitive.attributes.find("NORMAL");
					if (normIt == primitive.attributes.end())
						throw std::runtime_error("[GltfLoader] Mandatory NORMAL attribute missing!");
					auto& normAccessor = model.accessors[primitive.attributes["NORMAL"]];
					auto& normBufferView = model.bufferViews[normAccessor.bufferView];
					auto& normBuffer = model.buffers[normBufferView.buffer];
					assert(normAccessor.type == TINYGLTF_TYPE_VEC3 && "[GltfLoader] Unexpected type found for vertex normal!");
					assert(normAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && "[GltfLoader] Unexpected componentType found for vertex normal!");

					const uint8_t* normStart = normBuffer.data.data() + normBufferView.byteOffset + normAccessor.byteOffset;
					size_t normStride = normAccessor.ByteStride(normBufferView);
					assert(posAccessor.count == normAccessor.count && "[GltfLoader] Number of vertex positions and normals differ!");

					// Vertex UVs (optional)
					auto uvIt = primitive.attributes.find("TEXCOORD_0");
					const uint8_t* uvStart = nullptr;
					size_t uvStride = 0;
					if (uvIt != primitive.attributes.end())
					{
						auto& uvAccessor = model.accessors[primitive.attributes["TEXCOORD_0"]];
						auto& uvBufferView = model.bufferViews[uvAccessor.bufferView];
						auto& uvBuffer = model.buffers[uvBufferView.buffer];

						assert(uvAccessor.type == TINYGLTF_TYPE_VEC2 && "[GltfLoader] Unexpected type found for uv!");
						// TODO: add unsigned byte and unsigned short component type support
						assert(uvAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && "[GltfLoader] Unexpected componentType found for vertex normal!");

						uvStart = uvBuffer.data.data() + uvBufferView.byteOffset + uvAccessor.byteOffset;
						uvStride = uvAccessor.ByteStride(uvBufferView);
						assert(posAccessor.count == uvAccessor.count && "[GltfLoader] Number of vertex positions and uv differ!");
					}

					// Warn about unsupported attributes detected
					for (auto& [name, accessorIdx] : primitive.attributes)
						if (name != "POSITION" && name != "NORMAL" && name != "TEXCOORD_0")
							LOG("[GltfLoader] WARNING! Unsupported attributes found: " + name);	
					
					// Fill the vertex data
					// TODO: improve handling of optional attributes
					vertices.resize(posAccessor.count);
					if (uvStart)
					{
						for (size_t i = 0; i < posAccessor.count; i++)
						{
							const float* posPtr = reinterpret_cast<const float*>(posStart + i * posStride);
							vertices[i].pos = glm::vec3(posPtr[0], posPtr[1], posPtr[2]);
						
							const float* normPtr = reinterpret_cast<const float*>(normStart + i * normStride);
							vertices[i].normal = glm::vec3(normPtr[0], normPtr[1], normPtr[2]);

							const float* uvPtr = reinterpret_cast<const float*>(uvStart + i * uvStride);
							vertices[i].uv = glm::vec2(uvPtr[0], uvPtr[1]);
						}
					}
					else 
					{
						for (size_t i = 0; i < posAccessor.count; i++)
						{
							const float* posPtr = reinterpret_cast<const float*>(posStart + i * posStride);
							vertices[i].pos = glm::vec3(posPtr[0], posPtr[1], posPtr[2]);

							const float* normPtr = reinterpret_cast<const float*>(normStart + i * normStride);
							vertices[i].normal = glm::vec3(normPtr[0], normPtr[1], normPtr[2]);

							vertices[i].uv = glm::vec2(0.0f, 0.0f); // dummy UV
						}
					}

				}

				// TODO: properly handle loading meshes without indices, by calling the correct draw call
				assert(primitive.indices != -1 && "[GltfLoader] Indices are not the defined!");

				// Loading indices
				std::vector<uint32_t> indices;
				{
					auto& accessor = model.accessors[primitive.indices];
					auto& bufferView = model.bufferViews[accessor.bufferView];
					auto& buffer = model.buffers[bufferView.buffer];
					indices.resize(accessor.count);

					assert(accessor.type == TINYGLTF_TYPE_SCALAR && "[GltfLoader] Unexpected type found for indices!");
					assert(bufferView.byteStride == 0 && "[GltfLoader] Indices are not tightly packed!");
					
					// Infer the correct component type and read data
					switch (accessor.componentType)
					{
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
						{
							const uint8_t* raw = reinterpret_cast<const uint8_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
							for (size_t i = 0; i < indices.size(); i++)
								indices[i] = static_cast<uint32_t>(raw[i]);
							break;
						}

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
						{
							const uint16_t* raw = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
							for (size_t i = 0; i < indices.size(); i++)
								indices[i] = static_cast<uint32_t>(raw[i]);
							break;
						}

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
						{
							const uint32_t* raw = reinterpret_cast<const uint32_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
							for (size_t i = 0; i < indices.size(); i++)
								indices[i] = raw[i];
							break;
						}
						default:
							throw std::runtime_error("[GltfLoader] Unsupported index componentType!");
					}
				}

				// Retrieve material info
				MaterialID matId = rm.GetDefaultMaterial();
				if(primitive.material != -1)
					matId = materials.at(primitive.material);

				// Mesh name (to avoid conflicting names)
				std::string meshName{};
				if (j == 0)
					meshName = mesh.name;
				else
					meshName = mesh.name + " (" + std::to_string(j) + ')';


				// Create and load primitive as a mesh
				MeshID id = rm.LoadMesh(std::make_unique<Mesh>(vertices, indices, matId), meshName, renderer);
				primitives.push_back(id);

				j++; // increment primitive counter
			}
			meshes.insert(std::pair<int, std::vector<MeshID>>(static_cast<int>(i), primitives));
			i++;
		}
		LOG("[GltfLoader] Loaded " + std::to_string(meshes.size()) + " meshes");
	}

	// Load all textures in `model` and fill `textures` with the corresponding TextureIDs
	static void LoadTextures(tinygltf::Model& model, Renderer& renderer, std::unordered_map<int, TextureID>& textures)
	{
		if (model.textures.size() > MAX_TEXTURES)
			throw std::runtime_error("[GltfLoader] Tried to load "
				+ std::to_string(model.textures.size()) + " textures when MAX_TEXTURES is set to "
				+ std::to_string(MAX_TEXTURES) + "!"
			);

		auto& rm = ResourceManager::GetInstance();
		for (size_t i = 0; i < model.textures.size(); i++)
		{
			tinygltf::Texture& texture = model.textures[i];
			// texture.sampler <- currently ignored
			if (texture.source == -1)
				continue;

			tinygltf::Image& image = model.images[texture.source];

			// Check if image data wasn't loaded for some reasons
			// e.g. forget to put textures in the same path as the .glTF
			if (image.image.size() == 0)
				throw std::runtime_error("[GltfLoader] Image data missing! Check previous warnings or errors from the loader.");

			// Create texture object
			vk::ImageCreateInfo imageInfo {				
				.imageType = vk::ImageType::e2D,
				.format = vk::Format::eR8G8B8A8Srgb,
				.extent = vk::Extent3D{ static_cast<uint32_t>(image.width),static_cast<uint32_t>(image.height), 1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = vk::SampleCountFlagBits::e1,
				.tiling = vk::ImageTiling::eOptimal,
				.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
				.sharingMode = vk::SharingMode::eExclusive,
				.initialLayout = vk::ImageLayout::eUndefined
			};
			VmaAllocationCreateInfo allocInfo = { .usage = VMA_MEMORY_USAGE_AUTO };
			std::unique_ptr<Texture> tex = std::make_unique<Texture>(renderer.GetDevice(), imageInfo, allocInfo);

			// Load texture
			TextureID id = rm.LoadTexture(std::move(tex), texture.name, image.image.data(), image.image.size(), renderer);
			textures.insert(std::pair<int, TextureID>(static_cast<int>(i), id));
		}
		LOG("[GltfLoader] Loaded " + std::to_string(textures.size()) + " textures");
	}

	// Load all materials in `model` and fill `materials` with the corresponding MaterialIDs
	static void LoadMaterials(tinygltf::Model& model, std::unordered_map<int, TextureID>& textures, std::unordered_map<int, MaterialID>& materials)
	{
		if (model.materials.size() > MAX_MATERIALS)
			throw std::runtime_error("[GltfLoader] Tried to load "
				+ std::to_string(model.materials.size()) + " materials when MAX_MATERIALS is set to "
				+ std::to_string(MAX_MATERIALS) + "!"
			);

		auto& rm = ResourceManager::GetInstance();
		for (size_t i = 0; i < model.materials.size(); i++)
		{
			const auto& material = model.materials[i];

			// TODO: improve material system to PBR
			// NOTES on `baseColor`:
			// metal -> f0
			// dielectric -> albedo and f0 should be set to 0.04
			//               as a good approximation of dielectrics
			//               (see Real Time Rendering)
			auto baseColor = material.pbrMetallicRoughness.baseColorFactor;
			auto metalness = material.pbrMetallicRoughness.metallicFactor;
			auto roughness = material.pbrMetallicRoughness.roughnessFactor;
			float ambient = 0.02f;

			// According to the specs, baseColorTexture should be multiplied with baseColorFactor
			// Since Blender seems to export either one or the other this is left as TODO
			int albedoTexIndex = material.pbrMetallicRoughness.baseColorTexture.index;
			int metallicRoughnessTexIndex = material.pbrMetallicRoughness.metallicRoughnessTexture.index;

			std::unique_ptr<Material> mat = std::make_unique<Material>(
				glm::vec3(baseColor[0], baseColor[1], baseColor[2]),
				glm::vec4(0.0, roughness, metalness, 0.0),
				(albedoTexIndex == -1) ? -1 : textures[albedoTexIndex],
				(metallicRoughnessTexIndex == -1) ? -1 : textures[metallicRoughnessTexIndex]
			);
			
			MaterialID id = rm.LoadMaterial(std::move(mat), material.name);
			materials.insert(std::pair<int, MaterialID>(static_cast<int>(i), id));
		}
		LOG("[GltfLoader] Loaded " + std::to_string(materials.size()) + " materials");
	}

	static Light::Type ToLightType(const std::string& type)
	{
		if (type == "directional") return Light::Type::DIRECTIONAL;
		if (type == "point") return Light::Type::POINT;
		if (type == "spot") return Light::Type::SPOT;
	}

	// Create object and iterate recursively through its children
	static std::unique_ptr<Object> LoadNode(
		const tinygltf::Node& node, Object* parent, const tinygltf::Model& model,
		const std::unordered_map<int, std::vector<MeshID>>& lookUpMeshes, size_t& lightsCount
	)
	{
		// Object creation
		std::unique_ptr<Object> obj = nullptr;
		if (node.mesh != -1)
		{
			std::vector<MeshID> meshes = lookUpMeshes.at(node.mesh);
			obj = std::make_unique<Object>(node.name, meshes, parent);
		}
		else
		{
			obj = std::make_unique<Object>(node.name, parent);
		}

		// Loading light data (optional)
		if (node.light != -1)
		{
			// Retrieve light from glTF
			const tinygltf::Light& light = model.lights[node.light];

			// Set object light data
			Light lightData{
				ToLightType(light.type),							// type
				glm::vec3(											// color
					static_cast<float>(light.color[0]),
					static_cast<float>(light.color[1]),
					static_cast<float>(light.color[2])
				),
				static_cast<float>(light.intensity),				// intensity
				static_cast<float>(light.range),					// range
				static_cast<float>(cos(light.spot.innerConeAngle)),	// innerCone
				static_cast<float>(cos(light.spot.outerConeAngle)),	// outerCone
				light.name											// name
			};
			obj->SetLightData(lightData);
			
			// Increment the lights count
			lightsCount++;
		}

		// Apply transform
		// Transform -> see p.18 of glTF specs
		if (node.matrix.size() == 16) // if matrix is specified it will have priority
		{
			glm::mat4 mat = glm::make_mat4(node.matrix.data());
			obj->SetModelMatrix(mat);
		}
		else
		{
			if (node.scale.size())
				obj->SetScale(glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
			if (node.rotation.size()) // XYZW
				obj->SetRotation(glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2])); // WXYZ
			if (node.translation.size())
				obj->SetPosition(glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
		}
			
		// Iterate over its children
		for (const auto childIdx : node.children)
		{
			std::unique_ptr<Object> child = LoadNode(model.nodes[childIdx], obj.get(), model, lookUpMeshes, lightsCount);
			obj->AddChild(std::move(child)); // Move child object ownership to the parent
		}
		return std::move(obj);
	}

	// Parse filepath (either .glb or .gltf file) into model using tinygltf
	static void ParseFile(const std::filesystem::path& filepath, tinygltf::Model& model)
	{
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;
		const std::string extension = filepath.extension().string();

		// Check file extension and load file
		bool ret{};
		if (extension == ".glb")
			ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath.string());
		else if (extension == ".gltf")
			ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath.string());
		else
			throw std::runtime_error(
				"[GltfLoader] Tried to load unsupported file format.\n \
				 Currently supported file formats: .glb, .gltf."
			);

		if (!warn.empty())
			LOG("[GltfLoader] Warn: " + warn);
		if (!err.empty())
			throw std::runtime_error("[GltfLoader] Err: " + err);
		if (!ret)
			throw std::runtime_error("[GltfLoader] Failed to parse glTF: " + filepath.string());

		LOG("[GltfLoader] Parsed " + filepath.string());
	}

	// Load resources and setup `scene` with the data provided by the file in `filepath`
	// NOTE: 
	//    - `renderer` is used to call backend functions for loading resources
	//    - `filepath` must be a valid path to either a .gltf or .glb file, 
	//       otherwise an exception will be raised
	void LoadSceneFromGlTF(const std::filesystem::path& filepath, Scene& scene, Renderer& renderer)
	{
		// File parsing
		tinygltf::Model model;
		ParseFile(filepath, model); // Bottleneck D:

		// Load resources
		// NOTE: loading should respect the following order of dependencies
		// First you load textures -> materials -> meshes

		// LUT associating a glTF index with the corresponding ResourceID
		std::unordered_map<int, TextureID> textures;
		std::unordered_map<int, MaterialID> materials;
		std::unordered_map<int, std::vector<MeshID>> meshes;

		LoadTextures(model, renderer, textures);
		LoadMaterials(model, textures, materials);
		LoadMeshes(model, renderer, materials, meshes);

		// Check for max number of nodes
		if (model.nodes.size() > MAX_OBJECTS)
		{
			throw std::runtime_error("[GltfLoader] Tried to load "
				+ std::to_string(model.nodes.size()) + " objects when MAX_OBJECTS is set to "
				+ std::to_string(MAX_OBJECTS) + "!"
			);
		}

		// Iterate through each top-level node (parent = nullptr)
		size_t lightsCount = 0;
		for (const auto nodeIdx : model.scenes[model.defaultScene].nodes)
		{
			std::unique_ptr<Object> obj = LoadNode(model.nodes[nodeIdx], nullptr, model, meshes, lightsCount);
			scene.AddObject(std::move(obj)); // Move top-level object ownership to the scene
		}

		// Check for max number of lights
		if (lightsCount > MAX_LIGHTS)
		{
			throw std::runtime_error("[GltfLoader] Tried to load "
				+ std::to_string(lightsCount) + " lights when MAX_LIGHTS is set to "
				+ std::to_string(MAX_LIGHTS) + "!"
			);
		}
	}
}