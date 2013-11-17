#ifndef APPLICATION_H
#define APPLICATION_H

//#include "Timer.h"
#include <Camera.h>
#include <Renderer.h>
#include <InputController.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <windows.h>
#include <windowsx.h>
#include <string>
#include <exception>

////////////////////////////////////////////////////////////////////////////////////////////////////
#define win32Assert(resultHandle, errorMessage) \
	if (resultHandle == 0) \
	{ \
		MessageBox(NULL, #errorMessage, title.c_str(), MB_OK | MB_ICONEXCLAMATION); \
		dispose(); \
		exit(EXIT_FAILURE); \
	}

////////////////////////////////////////////////////////////////////////////////////////////////////
#define safePointerCall(pointer, call) \
	if (pointer) \
	{ \
		pointer->call; \
	}

////////////////////////////////////////////////////////////////////////////////////////////////////
LRESULT CALLBACK WindowProcedure(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

////////////////////////////////////////////////////////////////////////////////////////////////////
class Application
{
public:
	static const unsigned int BYTES_PER_PIXEL;
	static const glm::vec4 CLEAR_COLOR;
	static const glm::vec4 GLOBAL_AMBIENT_LIGHT;

	static Application* instance;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	Application(const std::string& title, unsigned int width, unsigned int height) :
		title(title),
		width(width),
		height(height),
		running(false),
		applicationHandle(0),
		windowHandle(0),
		deviceContextHandle(0),
		pixelFormat(0),
		openGLRenderingContextHandle(0),
		camera(0),
		renderer(0),
		inputController(0)
	{
		instance = this;
		create();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	~Application()
	{
		instance = 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void create()
	{
		applicationHandle = GetModuleHandle(0);
		win32Assert(RegisterClassEx(&createWindowClass()), "RegisterClassEx failed");
		win32Assert((windowHandle = CreateWindow(WINDOW_CLASS_NAME, title.c_str(), (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_CLIPSIBLINGS | WS_CLIPCHILDREN), CW_USEDEFAULT, CW_USEDEFAULT, width, height, NULL, NULL, applicationHandle, NULL)), "CreateWindow failed");
		win32Assert((deviceContextHandle = GetDC(windowHandle)), "GetDC() failed");
		win32Assert((pixelFormat = ChoosePixelFormat(deviceContextHandle, &PIXEL_FORMAT_DESCRIPTOR)), "ChoosePixelFormat() failed");
		win32Assert(SetPixelFormat(deviceContextHandle, pixelFormat, &PIXEL_FORMAT_DESCRIPTOR), "SetPixelFormat() failed");
		win32Assert((openGLRenderingContextHandle = wglCreateContext(deviceContextHandle)), "wglCreateContext() failed");
		win32Assert(wglMakeCurrent(deviceContextHandle, openGLRenderingContextHandle), "wglMakeCurrent() failed");
		ShowWindow(windowHandle, SW_SHOW);
		SetForegroundWindow(windowHandle);
		SetFocus(windowHandle);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setCamera(Camera& camera)
	{
		this->camera = &camera;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setInputController(InputController& inputController)
	{
		this->inputController = &inputController;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setRenderer(Renderer& renderer)
	{
		this->renderer = &renderer;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	int run()
	{
		try
		{
			safePointerCall(renderer, initialize());
		}

		catch (std::exception& e)
		{
			MessageBox(windowHandle, e.what(), title.c_str(), MB_OK | MB_ICONEXCLAMATION);
			dispose();
			exit(EXIT_FAILURE);
		}

		//Timer appTime;
		//appTime.start();
		double deltaTime = 0;
		running = true;
		MSG message;

		while (running)
		{
			if (PeekMessage(&message, NULL, 0, 0, PM_REMOVE))
			{
				if (message.message == WM_QUIT)
				{
					running = false;
				}

				else
				{
					TranslateMessage(&message);
					DispatchMessage(&message);
				}
			}

			else
			{
				//appTime.end();
				//deltaTime = appTime.getElapsedTime();
				deltaTime = 0.1;
				safePointerCall(inputController, swapBuffers());
				safePointerCall(inputController, update(deltaTime));
				safePointerCall(camera, update(deltaTime));

				try
				{
					safePointerCall(renderer, render(deltaTime));
					SwapBuffers(deviceContextHandle);
					//appTime.start();
				}

				catch (std::exception& e)
				{
					MessageBox(windowHandle, e.what(), title.c_str(), MB_OK | MB_ICONEXCLAMATION);
					DestroyWindow(windowHandle);
					running = false;
				}
			}
		}

		dispose();
		win32Assert(UnregisterClass(WINDOW_CLASS_NAME, applicationHandle), "UnregisterClass() failed");
		deviceContextHandle = 0;
		windowHandle = 0;
		applicationHandle = 0;
		return (int) message.wParam;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void halt()
	{
		DestroyWindow(windowHandle);
	}

	friend LRESULT CALLBACK WindowProcedure(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

private:
	static const char* WINDOW_CLASS_NAME;
	static const unsigned int COLOR_BUFFER_BITS;
	static const unsigned int DEPTH_BUFFER_BITS;
	static const unsigned int HAS_ALPHA;
	static const PIXELFORMATDESCRIPTOR PIXEL_FORMAT_DESCRIPTOR;

	std::string title;
	unsigned int width;
	unsigned int height;
	bool running;
	HINSTANCE applicationHandle;
	HWND windowHandle;
	HDC deviceContextHandle;
	int pixelFormat;
	HGLRC openGLRenderingContextHandle;
	Camera* camera;
	Renderer* renderer;
	InputController* inputController;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void dispose()
	{
		if (openGLRenderingContextHandle)
		{
			win32Assert(wglMakeCurrent(NULL, NULL), "wglMakeCurrent() failed");
			win32Assert(wglDeleteContext(openGLRenderingContextHandle), "wglDeleteContext() failed");
			openGLRenderingContextHandle = 0;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	WNDCLASSEX createWindowClass()
	{
		WNDCLASSEX windowClass;
		windowClass.cbSize = sizeof(WNDCLASSEX);
		windowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
		windowClass.lpfnWndProc = WindowProcedure;
		windowClass.cbClsExtra = 0;
		windowClass.cbWndExtra = 0;
		windowClass.hInstance = applicationHandle;
		windowClass.hIcon = LoadIcon(applicationHandle, MAKEINTRESOURCE(IDI_APPLICATION));
		windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
		windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
		windowClass.lpszMenuName = NULL;
		windowClass.lpszClassName = WINDOW_CLASS_NAME;
		windowClass.hIconSm	= LoadIcon(windowClass.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
		return windowClass;
	}

};

////////////////////////////////////////////////////////////////////////////////////////////////////
Application* Application::instance = 0;
const char* Application::WINDOW_CLASS_NAME = "windowClass";
const unsigned int Application::BYTES_PER_PIXEL = 4;
const unsigned int Application::COLOR_BUFFER_BITS = 32;
const unsigned int Application::DEPTH_BUFFER_BITS = 32;
const unsigned int Application::HAS_ALPHA = 0;
const PIXELFORMATDESCRIPTOR Application::PIXEL_FORMAT_DESCRIPTOR = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER, PFD_TYPE_RGBA, Application::COLOR_BUFFER_BITS, 0, 0, 0, 0, 0, 0, Application::HAS_ALPHA, 0, 0, 0, 0, 0, 0, Application::DEPTH_BUFFER_BITS, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0 };

////////////////////////////////////////////////////////////////////////////////////////////////////
LRESULT CALLBACK WindowProcedure(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int x, y;

	switch (message)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_KEYDOWN:
		safePointerCall(Application::instance->inputController, keyDown(wParam));
		break;

	case WM_KEYUP:
		safePointerCall(Application::instance->inputController, keyUp(wParam));
		break;

	case WM_LBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseButtonDown(MK_LBUTTON, x, y));
		break;

	case WM_MBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseButtonDown(MK_MBUTTON, x, y));
		break;

	case WM_RBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseButtonDown(MK_RBUTTON, x, y));
		break;

	case WM_LBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseButtonUp(MK_LBUTTON, x, y));
		break;

	case WM_MBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseButtonUp(MK_MBUTTON, x, y));
		break;

	case WM_RBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseButtonUp(MK_RBUTTON, x, y));
		break;

	case WM_MOUSEMOVE:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);
		safePointerCall(Application::instance->inputController, mouseMove(x, y));
		break;

	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
		break;
	}

	return 0;
}

#endif