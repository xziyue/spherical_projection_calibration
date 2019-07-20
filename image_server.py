from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
import glfw
import numpy as np
import platform as pyPlatform
import ctypes
from datetime import datetime

from gl_lib.transmat import *
from gl_lib.utility import *
from gl_lib.fps_camera import *
from gl_lib.gl_screenshot import save_screenshot_rgb, get_screenshot
import gl_lib.text_drawer
from shader import *

# import Pillow for loading images
from PIL import Image

windowSize = np.asarray((1200, 900), np.int)
windowBackgroundColor = (0.2, 0.2, 0.2, 1.0)
ambientColor = np.asarray((0.0, 0.0, 0.0), np.float32)
backColor = np.asarray((0.0, 0.0, 0.0), np.float32)
focalLength = 2.0
cameraWidth = 0.7
zNear = 0.1
zFar = 100.0

camera = FPSCamera()
camera.eyePos[:] = [ 3.52, 0.01, 3.50]
camera.set_pitch(0.0012)
camera.set_yaw(-2.35)
projector = FPSCamera()
projector.eyePos[2] = 4.0
controlId = 0
controlObjs = [camera, projector]
controlTexts = ['camera', 'projector']

vertices = np.asarray([
    -1.0, -1.0, 0.0,
    -1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0,
    1.0, 1.0, 0.0
], np.float32)


def debug_message_callback(source, msg_type, msg_id, severity, length, raw, user):
    msg = raw[0:length]
    print('debug', source, msg_type, msg_id, severity, msg)


# stores which keys are pressed and handle key press in the main loop
keyArray = np.array([False] * 300, np.bool)


def window_keypress_callback(theWindow, key, scanCode, action, mods):
    global useBicubic, controlId

    if key == glfw.KEY_UNKNOWN:
        return

    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            # respond escape here
            glfw.set_window_should_close(theWindow, True)
        elif key == glfw.KEY_P:
            # respond screenshot keypress
            nowTime = datetime.now()
            timeString = nowTime.strftime('%Y-%m-%d_%H:%M:%S')
            screenshotFmt = 'screenshot_{}.png'
            save_screenshot_rgb(screenshotFmt.format(timeString), windowSize)
        elif key == glfw.KEY_O:
            controlId = (controlId + 1) % len(controlObjs)
        else:
            keyArray[key] = True
    elif action == glfw.RELEASE:
        keyArray[key] = False


def keyboard_respond_func():
    global keyArray

    keyPressed = np.where(keyArray == True)
    for key in keyPressed[0]:
        if key in glfwKeyTranslator:
            controlObjs[controlId].respond_keypress(glfwKeyTranslator[key])


def window_resize_callback(theWindow, width, height):
    global windowSize
    windowSize = (width, height)
    glViewport(0, 0, width, height)


def window_cursor_callback(theWindow, xPos, yPos):
    global cursorPos
    xOffset = xPos - cursorPos[0]
    yOffset = yPos - cursorPos[1]
    controlObjs[controlId].respond_mouse_movement(xOffset, yOffset)
    cursorPos = (xPos, yPos)


def window_scroll_callback(theWindow, xOffset, yOffset):
    camera.respond_scroll(yOffset)


def create_uniform(programId, infos):
    result = dict()
    for name, tp in infos:
        uniform = GLUniform(programId, name, tp)
        result[name] = uniform
    return result


def get_camera_vectors(cam):
    height = cameraWidth / (windowSize[0] / windowSize[1])
    eyePos = cam.get_eye_pos()
    front = cam._get_front_dir()
    base = eyePos + focalLength * front - 0.5 * cameraWidth * cam._get_right_dir() \
           - 0.5 * height * cam._get_up_dir()
    x = cameraWidth * cam._get_right_dir()
    y = height * cam._get_up_dir()

    return eyePos, base, x, y

def create_texture(image):
    assert image.dtype == np.uint8
    assert len(image.shape) == 3

    # unpack alignment first
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    texture = GLTexture2D()
    texture.bind()

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        image.shape[1],
        image.shape[0],
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        get_numpy_unit8_array_pointer(image)
    )

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    # in this case, GL_NEAREST and GL_LINEAR should make little difference
    # because only the center of pixels are used in the shader
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    texture.unbind()

    return texture


def image_server(evtQueue, resultQueue):

    # resource-taking objects
    resObjs = []

    # initialize glfw
    glfw.init()

    # set glfw config
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, GL_FALSE)

    if pyPlatform.system().lower() == 'darwin':
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # create window
    theWindow = glfw.create_window(windowSize[0], windowSize[1], 'Spherical Projection', None, None)
    # make window the current context
    glfw.make_context_current(theWindow)


    # enable z-buffer
    glEnable(GL_DEPTH_TEST)

    # set resizing callback function
    # glfw.set_framebuffer_size_callback(theWindow, window_resize_callback)

    #glfw.set_key_callback(theWindow, window_keypress_callback)
    # disable cursor
    #glfw.set_input_mode(theWindow, glfw.CURSOR, glfw.CURSOR_DISABLED)

    #glfw.set_cursor_pos_callback(theWindow, window_cursor_callback)
    # initialize cursor position
    cursorPos = glfw.get_cursor_pos(theWindow)

    # glfw.set_scroll_callback(theWindow, window_scroll_callback)

    vbo = VBO(vertices, 'GL_STATIC_DRAW')
    vbo.create_buffers()
    resObjs.append(vbo)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo.bind()
    vbo.copy_data()
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    # compile program
    renderProgram = GLProgram(rayTracingVertexShaderSource, rayTracingFragmentShaderSource)
    renderProgram.compile_and_link()
    uniformInfos = [
        ('backColor', 'vec3f'),
        ('ambientColor', 'vec3f'),
        ('o_c', 'vec3f'),
        ('o_p', 'vec3f'),
        ('x_c', 'vec3f'),
        ('y_c', 'vec3f'),
        ('x_p', 'vec3f'),
        ('y_p', 'vec3f'),
        ('c_c', 'vec3f'),
        ('c_p', 'vec3f'),
        ('winSize', 'vec2f')
    ]
    uniforms = create_uniform(renderProgram.get_program_id(), uniformInfos)


    # keep rendering until the window should be closed
    while not glfw.window_should_close(theWindow):

        # set background color
        glClearColor(*windowBackgroundColor)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        renderProgram.use()

        # update uniforms
        o_c, c_c, x_c, y_c = get_camera_vectors(camera)
        o_p, c_p, x_p, y_p = get_camera_vectors(projector)
        uniforms['o_c'].update(o_c)
        uniforms['x_c'].update(x_c)
        uniforms['y_c'].update(y_c)
        uniforms['c_c'].update(c_c)

        uniforms['o_p'].update(o_p)
        uniforms['x_p'].update(x_p)
        uniforms['y_p'].update(y_p)
        uniforms['c_p'].update(c_p)
        uniforms['backColor'].update(backColor)
        uniforms['ambientColor'].update(ambientColor)
        uniforms['winSize'].update(windowSize.astype(np.float32))

        try:
            newImage = evtQueue.get(timeout=0.05)
        except Exception as e:
            # tell glfw to poll and process window events
            glfw.poll_events()
            # swap frame buffer
            glfw.swap_buffers(theWindow)
            continue

        texture = create_texture(newImage)

        glBindVertexArray(vao)
        glActiveTexture(GL_TEXTURE0)
        texture.bind()
        glDrawArrays(GL_TRIANGLES, 0, 6)
        texture.unbind()
        glBindVertexArray(0)

        texture.delete()

        # respond key press
        keyboard_respond_func()
        # tell glfw to poll and process window events
        glfw.poll_events()
        # swap frame buffer
        glfw.swap_buffers(theWindow)

        result = get_screenshot(windowSize)
        resultQueue.put(result)

    for obj in resObjs:
        obj.delete()

    # terminate glfw
    glfw.terminate()

