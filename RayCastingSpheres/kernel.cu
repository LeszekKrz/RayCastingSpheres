#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"
#include "functions.hpp"
#include "cudaFunctions.cuh"

#include <stdio.h>
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846


// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);


__global__ void fillKernel(char* table, int width)
{
    char* m_table = table + blockIdx.x * width * 3 + blockIdx.y * 3;
    *m_table = 0;
    *(m_table + 1) = 255;
    *(m_table + 2) = 0;
}
__global__ void fillKernel2(char* table, int width)
{
    char* m_table = table + blockIdx.x * width * 3 + blockIdx.y * 3;
    *m_table = 0;
    *(m_table + 1) = 0;
    *(m_table + 2) = 255;
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int size = SCR_WIDTH * SCR_HEIGHT;

    // Prepare texture buffers
    unsigned char* h_texture;
    unsigned char* d_texture;
    h_texture = (unsigned char*)malloc(size * 3);
    PrepareTexture(&d_texture, size * 3);



    // Generate citcles anf lights
    circles h_circles, d_circles;
    h_circles.n = 1000;
    CreateCircles(&h_circles);
    PrepareCircles(h_circles, &d_circles);
    //DisplayCircles(h_circles);

    lights h_lights, d_lights;
    h_lights.n = 5;
    CreateLights(&h_lights);
    PrepareLights(h_lights, &d_lights);
    //DisplayLights(h_lights);


    // Prepare the camera
    camera h_camera;
    h_camera.pos = make_float3(0, 0, -200);
    h_camera.width = 800;
    h_camera.height = 600;
    h_camera.fovH = 80;
    h_camera.fovV = 60;
    PrepareCamera(&h_camera);

    scene d_scene{ d_circles, d_lights, h_camera };
    scene h_scene{ h_circles, h_lights, h_camera };
    


    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    Shader ourShader("VertexShader.txt", "FragmentShader.txt");

    float vertices[] = {
        // positions         // colors
         1.0f, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
         1.0f, -1.0f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,
         -1.0f,  -1.0f, 0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
         -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f

    };
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    unsigned char test = 1;
    unsigned int test2 = test << 10;

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    CopyTexture(&h_texture, &d_texture, size * 3, true);
    rayTrace(d_scene, d_texture);
    CopyTexture(&h_texture, &d_texture, size * 3, false);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, h_texture);

    // render loop
    // -----------
     
    float step = PI / 360;
    float angle = 0;

    int j = 0;

    double last_time = glfwGetTime() - 50;
    double maxFps = 0;

    while (!glfwWindowShouldClose(window))
    {
        double current_time = glfwGetTime();
        if (maxFps < 1 / (current_time - last_time)) maxFps = 1 / (current_time - last_time);
        last_time = current_time;
        

        processInput(window);
        d_scene._camera.pos = make_float3(cos(angle) * 200, 0, sin(angle) * 200);
        h_scene._camera.pos = make_float3(cos(angle) * 200, 0, sin(angle) * 200);
        angle += step;
        PrepareCamera(&d_scene._camera);
        rayTrace(d_scene, d_texture);
        CopyTexture(&h_texture, &d_texture, size * 3, false);

        // Prepared for CPU computting
        //rayTraceCPU(h_scene, h_texture);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, h_texture);

        ourShader.use();

        // render the triangle
        glBindVertexArray(VAO);
        //glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
        j++;
        if (j == 100) std::cout << maxFps << std::endl;
    }

    

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();


    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}