#version 400

layout(location = 0) out vec4 oColor;

uniform sampler2D uFontTex;
uniform vec4 uColor;

in vec2 fUv;

void main()
{
	float a = texture(uFontTex, fUv).x;
	oColor = vec4(uColor.rgb, a);
}