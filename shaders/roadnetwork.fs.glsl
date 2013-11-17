#version 400

in vec2 fUv;
in vec3 fLightDir;
in vec3 fViewDir;

layout(location = 0) out vec4 oColor;

uniform vec4 uLightColor;
uniform float uLightIntensity;
uniform vec4 uAmbientColor;
uniform vec4 uDiffuseColor;
uniform vec4 uSpecularColor;
uniform float uShininess;
uniform sampler2D uBaseTexture;
uniform sampler2D uBumpTexture;

vec3 unpackNormal(vec4 packedNormal)
{
	return packedNormal.xyz * 2.0 - 1.0;
}

void main()
{
	vec2 uv = fUv;

	vec4 texelColor = texture(uBaseTexture, uv);
	vec3 normal = unpackNormal(texture(uBumpTexture, uv));
	vec4 color = uDiffuseColor * texelColor;

	float diffuseAttenuation = max(dot(normal, fLightDir.xyz), 0.0);
	
	float specularAttenuation = 0.0;
	if (diffuseAttenuation > 0.0)
	{
		vec3 lightReflection = normalize(reflect(-fLightDir.xyz, normal));
		specularAttenuation = pow(max(dot(lightReflection, fViewDir), 0.0), uShininess);
	}

	float selfShadowing = step(0.1, fLightDir.z);

	oColor = uAmbientColor + (uLightIntensity * uLightColor * diffuseAttenuation * color + specularAttenuation * uSpecularColor * selfShadowing);
}
