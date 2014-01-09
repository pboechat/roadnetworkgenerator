#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

/********************************
 *	VECTOR MATH LIBRARY WRAPPER
 ********************************/


#ifdef USE_GLM

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace glm_utils
{

inline static glm::vec2 rotate2D(const glm::vec2& vec, float rad)
{
	return glm::vec2((vec.x * cos(rad)) - (vec.y * sin(rad)), (vec.y * cos(rad)) +  (vec.x * sin(rad)));
}

inline static float dotPerp(const glm::vec2& v0, const glm::vec2& v1)
{
	return v0.x * v1.y - v0.y * v1.x;
}

}

typedef glm::mat4 vml_mat4;
typedef glm::mat3 vml_mat3;
typedef glm::vec4 vml_vec4;
typedef glm::vec3 vml_vec3;
typedef glm::vec2 vml_vec2;
typedef glm::quat vml_quat;

//#define vml_mat4 glm::mat4
//#define vml_mat3 glm::mat3
//#define vml_vec4 glm::vec4
//#define vml_vec3 glm::vec3
//#define vml_vec2 glm::vec2
//#define vml_quat glm::quat

#define vml_rotate2D(vec, rad) glm_utils::rotate2D(vec, rad)
#define vml_rotate glm::rotate
#define vml_radians glm::radians
#define vml_distance glm::distance
#define vml_normalize glm::normalize
#define vml_dot glm::dot
#define vml_dot_perp(_v0, _v1) glm_utils::dotPerp(_v0, _v1)
#define vml_length glm::length
#define vml_cross glm::cross
#define vml_to_quat glm::toQuat
#define vml_to_mat4 glm::toMat4
#define vml_scale glm::scale
#define vml_angle_axis glm::angleAxis
#define vml_perspective glm::perspective
#define vml_look_at glm::lookAt

#endif

#endif