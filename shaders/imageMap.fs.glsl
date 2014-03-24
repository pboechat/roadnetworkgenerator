#version 400

subroutine vec4 shadingModelType();

subroutine uniform shadingModelType shadingModel;

layout(location = 0) out vec4 oColor;

uniform sampler2D uBaseTex;
uniform vec4 uColor1;
uniform vec4 uColor2;

in vec2 fUv;

subroutine(shadingModelType) vec4 flatColor()
{
	float a = texture(uBaseTex, fUv).x;
	if (a == 0.0) 
		return uColor1;
	else
		return uColor2;
}

subroutine(shadingModelType) vec4 luminance()
{
	float a = texture(uBaseTex, fUv).x;
	return mix(uColor1, uColor2, a);
}

void main()
{
	oColor = shadingModel();
}