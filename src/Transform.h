#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector_math.h>

struct Transform
{
	vml_vec3 scale;
	vml_quat rotation;
	vml_vec3 position;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	Transform()
	{
		scale = vml_vec3(1, 1, 1);
	}
	Transform(const vml_vec3& scale, const vml_quat& rotation, const vml_vec3& position) : scale(scale), rotation(rotation), position(position) {}
	~Transform() {}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_vec3 up() const
	{
		return rotation * vml_vec3(0, 1, 0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_vec3 right() const
	{
		return rotation * vml_vec3(1, 0, 0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_vec3 forward() const
	{
		return rotation * vml_vec3(0, 0, -1);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_mat4 toMat4() const
	{
		vml_mat4 model = vml_to_mat4(rotation);
		model[0][3] = position.x;
		model[1][3] = position.y;
		model[2][3] = position.z;
		return vml_scale(model, scale);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void lookAt(const vml_vec3& forward, const vml_vec3& up = vml_vec3(0, 1, 0))
	{
		vml_vec3 z = vml_normalize(position - forward);
		vml_vec3 x = vml_normalize(vml_cross(up, z));
		vml_vec3 y = vml_normalize(vml_cross(z, x));
		rotation = vml_to_quat(vml_mat3(x.x, x.y, x.z,
										y.x, y.y, y.z,
										z.x, z.y, z.z));
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_vec3 operator * (const vml_vec3& other) const
	{
		vml_vec3 vector = other;
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