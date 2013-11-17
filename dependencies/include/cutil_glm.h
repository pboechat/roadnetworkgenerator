#ifndef CUTIL_GLM_H
#define CUTIL_GLM_H

#pragma once

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#define vec3FieldDeclaration(name) \
	protected: \
		float x_##name, y_##name, z_##name; \
	public: \
		inline __device__ __host__ glm::vec3 get##name() const { return glm::vec3(x_##name, y_##name, z_##name); } \
		inline __device__ __host__ void set##name(const glm::vec3& arg0) { x_##name = arg0.x; y_##name = arg0.y; z_##name = arg0.z; }

#define quatFieldDeclaration(name) \
	protected: \
		float x_##name, y_##name, z_##name, w_##name; \
	public: \
		inline __device__ __host__ glm::quat get##name() const { return glm::quat(w_##name, x_##name, y_##name, z_##name); } \
		inline __device__ __host__ void set##name(const glm::quat& arg0) { x_##name = arg0.x; y_##name = arg0.y; z_##name = arg0.z; w_##name = arg0.w; }

#define mat4x3FieldDeclaration(name) \
	protected: \
		float	m00_##name, m01_##name, m02_##name, \
				m10_##name, m11_##name, m12_##name, \
				m20_##name, m21_##name, m22_##name, \
				m30_##name, m31_##name, m32_##name; \
	public: \
		inline __device__ __host__ glm::mat4x3 get##name() const { return glm::mat4x3(m00_##name, m01_##name, m02_##name, \
																				      m10_##name, m11_##name, m12_##name, \
																					  m20_##name, m21_##name, m22_##name, \
																					  m30_##name, m31_##name, m32_##name); }

#endif