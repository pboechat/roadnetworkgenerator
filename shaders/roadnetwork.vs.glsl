#version 400

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUv;
layout(location = 3) in vec4 vTangent;

out vec2 fUv;
out vec3 fLightDir;
out vec3 fViewDir;

uniform mat4 uView;
uniform mat4 uViewInverse;
uniform mat4 uViewInverseTranspose;
uniform mat4 uViewProjection;
uniform vec4 uLightDir;

void main()
{
	vec3 normal = normalize((uViewInverseTranspose * vec4(vNormal, 0.0)).xyz);
	vec3 tangent = normalize((uViewInverseTranspose * vTangent).xyz);
	vec3 binormal = normalize(cross(normal, tangent) * vTangent.w);

	mat3 tbn = transpose(mat3(tangent, binormal, normal));

	fUv = vUv;
	fLightDir = tbn * normalize(uLightDir.xyz);
	fViewDir = tbn * -normalize(vPosition.xyz);

	gl_Position = uViewProjection * vPosition;
}

