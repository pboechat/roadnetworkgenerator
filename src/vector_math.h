#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

/********************************
 *	VECTOR MATH LIBRARY WRAPPER
 ********************************/

#ifdef USE_GLM

//////////////////////////////////////////////////////////////////////////
//	GLM
//////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace vml_glm
{

inline static glm::vec2 rotate2D(const glm::vec2& vec, float rad)
{
	return glm::vec2((vec.x * cos(rad)) - (vec.y * sin(rad)), (vec.y * cos(rad)) +  (vec.x * sin(rad)));
}

inline static float dotPerp(const glm::vec2& v0, const glm::vec2& v1)
{
	return v0.x * v1.y - v0.y * v1.x;
}

inline static float angle(const glm::vec2& v0, const glm::vec2& v1)
{
	return glm::acos(glm::dot(v0, v1) / (glm::length(v0) * glm::length(v1)));
}

inline static glm::vec2 perp(const glm::vec2& v)
{
	return glm::vec2(v.y, -v.x);
}

}

typedef glm::mat4x3 vml_mat4x3;
typedef glm::mat4 vml_mat4;
typedef glm::mat3 vml_mat3;
typedef glm::mat3 vml_mat2;
typedef glm::vec4 vml_vec4;
typedef glm::vec3 vml_vec3;
typedef glm::vec2 vml_vec2;
typedef glm::quat vml_quat;

#define vml_rotate2D(vec, rad) vml_glm::rotate2D(vec, rad)
#define vml_rotate glm::rotate
#define vml_translate glm::translate
#define vml_radians glm::radians
#define vml_distance glm::distance
#define vml_normalize glm::normalize
#define vml_dot glm::dot
#define vml_angle(vec1, vec2) vml_glm::angle(vec1, vec2)
#define vml_perp(vec1) vml_glm::perp(vec1)
#define vml_dot_perp(_v0, _v1) vml_glm::dotPerp(_v0, _v1)
#define vml_length glm::length
#define vml_cross glm::cross
#define vml_to_quat glm::toQuat
#define vml_to_mat4 glm::toMat4
#define vml_scale glm::scale
#define vml_angle_axis glm::angleAxis
#define vml_perspective glm::perspective
#define vml_look_at glm::lookAt
#define vml_mix glm::mix

#endif

#define vec2FieldDeclaration(name, decl) \
	protected: \
		float x_##name, y_##name; \
	public: \
		inline decl vml_vec2 get##name() const { return vml_vec2(x_##name, y_##name); } \
		inline decl void set##name(const vml_vec2& arg0) { x_##name = arg0.x; y_##name = arg0.y; }

#define vec3FieldDeclaration(name, decl) \
	protected: \
		float x_##name, y_##name, z_##name; \
	public: \
		inline decl vml_vec3 get##name() const { return vml_vec3(x_##name, y_##name, z_##name); } \
		inline decl void set##name(const vml_vec3& arg0) { x_##name = arg0.x; y_##name = arg0.y; z_##name = arg0.z; }

#define quatFieldDeclaration(name) \
	protected: \
		float x_##name, y_##name, z_##name, w_##name; \
	public: \
		inline decl vml_quat get##name() const { return vml_quat(w_##name, x_##name, y_##name, z_##name); } \
		inline decl void set##name(const vml_quat& arg0) { x_##name = arg0.x; y_##name = arg0.y; z_##name = arg0.z; w_##name = arg0.w; }

#define mat4x3FieldDeclaration(name) \
	protected: \
		float	m00_##name, m01_##name, m02_##name, \
				m10_##name, m11_##name, m12_##name, \
				m20_##name, m21_##name, m22_##name, \
				m30_##name, m31_##name, m32_##name; \
	public: \
		inline decl vml_mat4x3 get##name() const { \
			return vml_mat4x3(	m00_##name, m01_##name, m02_##name, \
								m10_##name, m11_##name, m12_##name, \
								m20_##name, m21_##name, m22_##name, \
								m30_##name, m31_##name, m32_##name); \
		}

#endif