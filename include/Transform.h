#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Transform
{
	glm::vec3 scale;
	glm::quat rotation;
	glm::vec3 position;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	Transform()
	{
		scale = glm::vec3(1, 1, 1);
	}
	Transform(const glm::vec3& scale, const glm::quat& rotation, const glm::vec3& position) : scale(scale), rotation(rotation), position(position) {}
	~Transform() {}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::vec3 up() const
	{
		return rotation * glm::vec3(0, 1, 0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::vec3 right() const
	{
		return rotation * glm::vec3(1, 0, 0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::vec3 forward() const
	{
		return rotation * glm::vec3(0, 0, -1);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::mat4 toMat4() const
	{
		glm::mat4 model = glm::toMat4(rotation);
		model[0][3] = position.x;
		model[1][3] = position.y;
		model[2][3] = position.z;
		return glm::scale(model, scale);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void lookAt(const glm::vec3& forward, const glm::vec3& up = glm::vec3(0, 1, 0))
	{
		glm::vec3 z = glm::normalize(position - forward);
		glm::vec3 x = glm::normalize(glm::cross(up, z));
		glm::vec3 y = glm::normalize(glm::cross(z, x));
		rotation = glm::toQuat(glm::mat3(x.x, x.y, x.z,
										 y.x, y.y, y.z,
										 z.x, z.y, z.z));
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::vec3 operator * (const glm::vec3& other) const
	{
		glm::vec3 vector = other;
		vector = scale * vector;
		vector = rotation * vector;
		vector += position;
		return vector;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline Transform operator * (const Transform& other) const
	{
		return Transform(other.scale, rotation * other.rotation, position + other.position);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline Transform& operator *= (const Transform& other)
	{
		scale = other.scale;
		rotation = other.rotation * rotation;
		position += other.position;
		return *this;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline Transform& operator = (const Transform& other)
	{
		this->scale = other.scale;
		this->rotation = other.rotation;
		this->position = other.position;
		return *this;
	}

};

#endif