#include <iso646.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <tgmath.h>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define SAMPLE_SIZE 10

#define VIEWPORT_WIDTH 2.0
#define VIEWPORT_HEIGHT ((VIEWPORT_WIDTH * IMAGE_HEIGHT)/IMAGE_WIDTH)

#define cahr char

struct vec3 {double x, y, z;};

double vec_dist(struct vec3 p1, struct vec3 p2) {
    return sqrt(((p2.x-p1.x) * (p2.x-p1.x)) + ((p2.y-p1.y) * (p2.y-p1.y)) + ((p2.z-p1.z) * (p2.z-p1.z)));
}

double vec_len(struct vec3 p1) {
    return sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
}

double vec_dot(struct vec3 v1, struct vec3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

struct vec3 vec_mul(struct vec3 v1, struct vec3 v2) {
    return (struct vec3) {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}

struct vec3 vec_div(struct vec3 v1, struct vec3 v2) {
    return (struct vec3) {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};
}

struct vec3 vec_add(struct vec3 v1, struct vec3 v2) {
    return (struct vec3) {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

struct ray {struct vec3 p1, p2;};

double sq(double x) {
    return x * x;
}

struct vec3 vec_sub(struct vec3 p1, struct vec3 p2) {
    return (struct vec3) {(p2.x - p1.x), (p2.y - p1.y), (p2.z - p1.z)};
}

struct normal_ray {struct vec3 sphere_point; struct vec3 normal; double t; bool outside_face; bool exists; struct vec3 color;};

struct sphere {struct vec3 center; double radius; struct vec3 color;};

struct normal_ray intersect_t(struct sphere *spheres, int length, struct ray ray) {
    double closest_t = -1.0;
    double best_x,best_y,best_z;
    struct vec3 best_normal_return;
    bool best_outside;
    bool exists = false;
    struct vec3 color;
    // modify to work with sphere
    for (int i = 0; i < length; i++) {
        double t = -1.0;
        double a = vec_dot(ray.p2, ray.p2) + vec_dot(ray.p1, ray.p1) - 2*vec_dot(ray.p1, ray.p2);
        double b = 2 * vec_dot(vec_sub(ray.p1, spheres[i].center), vec_sub(ray.p2, ray.p1));
        double c = -sq(spheres[i].radius) + vec_dot(ray.p1, ray.p1) + vec_dot(spheres[i].center, spheres[i].center) - 2 * vec_dot(ray.p1, spheres[i].center);
        // x = x1+t(x2-x1)
        // y = y1+t(y2-y1)
        // z = z1+t(z2-z1)

        if ((sq(b) - 4 * a * c) < 0) {
            continue;
        }

        double abcminus = (-b - sqrt(sq(b) - 4 * a * c))/(2 * a);
        double abcplus = (-b + sqrt(sq(b) - 4 * a * c))/(2 * a);

        double smol = 0.00000001;
        if (abcplus < smol && abcminus < smol) {
            continue;
        } else if (abcminus < smol) {
            t = abcplus;
        } else if (abcplus < smol) {
            t = abcminus;
        } else if (abcminus < abcplus) {
            t = abcminus;
        } else {
            t = abcplus;
        }

        double x = ray.p1.x + t * (ray.p2.x - ray.p1.x);
        double y = ray.p1.y + t * (ray.p2.y - ray.p1.y);
        double z = ray.p1.z + t * (ray.p2.z - ray.p1.z);
        struct vec3 outward_normal = {
            (x - spheres[i].center.x)/spheres[i].radius,
            (y - spheres[i].center.y)/spheres[i].radius,
            (z - spheres[i].center.z)/spheres[i].radius
        };

        // if (fabs(1.0 - vec_len(outward_normal)) > 5.0) {
        //     printf("outward_normal is x( %lf ", vec_len(outward_normal));
        // }
        struct vec3 normal_return;
        double vec_check = vec_dot(vec_sub(ray.p2,ray.p1), outward_normal);
        bool outside;
        if (vec_check >= 0.0) { // outside
            normal_return = outward_normal;
            outside = true;
        } else { // not
            normal_return = vec_mul((struct vec3) {-1.0,-1.0,-1.0}, outward_normal);
            outside = false;
        }
        if (closest_t == -1.0 || (t != -1.0 && t < closest_t)) { // better check
            closest_t = t;
            exists = true;
            best_x = x;
            best_y = y;
            best_z = z;
            best_normal_return = normal_return;
            best_outside = outside;
            color = spheres[i].color;
        }
    }
    return (struct normal_ray) { .sphere_point = {best_x,best_y,best_z}, .normal = best_normal_return, .t = closest_t, .outside_face = best_outside, .exists = exists, .color = color };
}

struct vec3 back_map(double px, double py) {
    // assume viewport centered at origin and in xy plane
    return (struct vec3) {-1.0 * ((px/(double)IMAGE_WIDTH * (double) VIEWPORT_WIDTH) - VIEWPORT_WIDTH/2.0),-1.0 * ((py/(double) IMAGE_HEIGHT * (double) VIEWPORT_HEIGHT) - VIEWPORT_HEIGHT/2.0), 0.0};
}



/* TODO: optimize
TAG: line 131
TAG: line 140
*/
struct vec3 random_sphere_generator() { // TODO: optimize
    while (true) {
        double offset_x = (double)rand()/(RAND_MAX/2) - 1;
        double offset_y = (double)rand()/(RAND_MAX/2) - 1;
        double offset_z =  (double)rand()/(RAND_MAX/2) - 1;
        struct vec3 offset = {offset_x, offset_y, offset_z};
        double len = vec_len(offset);
        if (len < 1) {
            return vec_div(offset,(struct vec3) {len,len,len});
        }
    }
}

struct vec3 ray_color(struct ray ray, int depth) {
    if (depth > 10) {
        return (struct vec3) { 0.0, 0.0, 0.0 };
    }
    struct sphere spheres[] = {
        {.center = {0.0, 5.0, 20.0}, .radius = 10, .color = {0.1,0.5,0.1}},
        {.center = {0.0, -1005.0,20.0}, .radius = 1000, .color = {0.1,0.1,0.5}}
    };
    struct normal_ray normal_from_point = intersect_t(spheres, 2, ray);
    if (normal_from_point.exists) {
        struct vec3 center_of_reflection = vec_add(normal_from_point.normal, normal_from_point.sphere_point);
        struct vec3 reflection_offset = vec_add(center_of_reflection, random_sphere_generator());
        struct ray reflected_ray = {normal_from_point.sphere_point, reflection_offset};
        return vec_mul(normal_from_point.color, ray_color(reflected_ray, depth+1));
    } else {
        // return (struct vec3) { 0.61176470588, 0.76470588235, 0.90196078431 }; // Background color!
        return (struct vec3) { 1.0, 1.0, 1.0 };
    }
}

struct vec3 pixel_color(int px, int py, int sample_size) { //make use sample sizes
    // (30, 50) (29.5..30.5, 49.5..50.5)
    // randomize px and py
    // rand() = random int between 0 and rand_max
    // (rand()/RAND_MAX)-0.5
    struct vec3 focal_point = {0.0,0.0,1.0};
    struct vec3 sum = {0.0,0.0,0.0};
    for (int i = 0; i < sample_size; i++) {
        double anti_aliasing_offset = ((double) rand()/RAND_MAX) - 0.5;
        sum = vec_add(sum, ray_color((struct ray){back_map(px + anti_aliasing_offset, py + anti_aliasing_offset), focal_point}, 0));
    }
    return vec_div(sum,(struct vec3) {(double) sample_size,(double) sample_size,(double) sample_size});
}

int main(void) {
    FILE *fptr = fopen("blah.ppm", "w");
    if (!fptr) {
        perror("Failed to open file");
        return 1;
    }
    // Write the PPM header
    fprintf(fptr, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    // Generate the gradient
    for (int py = IMAGE_HEIGHT-1; py >= 0; py--) {  // Iterate over height first
        for (int px = 0; px < IMAGE_WIDTH; px++) {  // Iterate over width
            struct vec3 color = pixel_color(px, py, SAMPLE_SIZE);
            fprintf(fptr, "%d %d %d ", (int)(sqrt(color.x)*255.0), (int)(sqrt(color.y)*255.0), (int)(sqrt(color.z)*255.0));
        }
        fprintf(fptr, "\n"); // Ensure each row starts on a new line
    }
    fclose(fptr);
    printf("PPM file generated: blah.ppm\n");
    return 0;
}
