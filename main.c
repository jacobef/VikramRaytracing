#include <iso646.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <tgmath.h>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define SAMPLE_SIZE 100

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

struct vec3 vec_mul_scalar(double scalar, struct vec3 vec) {
    return (struct vec3) { vec.x*scalar, vec.y*scalar, vec.z*scalar };
}

struct vec3 vec_div(struct vec3 v1, struct vec3 v2) {
    return (struct vec3) {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};
}

struct vec3 vec_div_scalar(struct vec3 vec, double scalar) {
    return (struct vec3) { vec.x/scalar, vec.y/scalar, vec.z/scalar };
}

struct vec3 vec_add(struct vec3 v1, struct vec3 v2) {
    return (struct vec3) {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

struct vec3 normalize(struct vec3 vec) {
    return vec_div_scalar(vec, vec_len(vec));
}

struct ray {struct vec3 p1, p2;};

struct vec3 ray_to_vec(struct ray ray) {
    return (struct vec3) { ray.p2.x - ray.p1.x, ray.p2.y - ray.p1.y, ray.p2.z - ray.p1.z };
}

double sq(double x) {
    return x * x;
}

struct vec3 vec_sub(struct vec3 p1, struct vec3 p2) {
    return (struct vec3) {(p1.x - p2.x), (p1.y - p2.y), (p1.z - p2.z)};
}

struct Dielectric {double refraction_index;};

struct Metal {struct vec3 albedo; double fuzz;};

struct Diffuse {struct vec3 color;};

union materialUnion {
    struct Diffuse diffuse;
    struct Metal metal;
    struct Dielectric dielectric;
};

enum materialTag {
    DIFFUSE,
    METAL,
    DIELECTRIC,
};

struct material {
    union materialUnion data;
    enum materialTag tag;
};

struct sphere {struct vec3 center; double radius; struct vec3 color; struct material material;};

struct normal_ray {struct vec3 sphere_point; struct vec3 normal; double t; bool outside_face; bool exists; struct material material;};

// Either:
// - Diffuse, with vec3 color
// Or:
// - Metal, with vec3 albedo and double fuzz

struct vec3 refract(struct vec3 ray_in, struct vec3 normal, double etai_over_etat) {
    double cos_theta = fmin(vec_dot(vec_mul_scalar(-1.0,ray_in), normal), 1.0);
    struct vec3 r_out_perp = vec_mul_scalar(etai_over_etat, vec_add(ray_in, vec_mul_scalar(cos_theta, normal)));
    struct vec3 r_out_parallel = vec_mul_scalar(-1.0, vec_mul_scalar(sqrt(fabs(1.0 - sq(vec_len(r_out_perp)))), normal));
    return vec_add(r_out_perp, r_out_parallel);
}

struct vec3 reflect(struct vec3 ray_in, struct vec3 normal) {
    return vec_sub(ray_in, vec_mul_scalar(2.0, vec_mul_scalar(vec_dot(ray_in, normal), normal)));
}

double reflectance(double cosine, double refraction_index) {
    double r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1-r0) * pow((1-cosine), 5);
}


struct normal_ray intersect_t(struct sphere *spheres, int length, struct ray ray) {
    double closest_t = -1.0;
    double best_x,best_y,best_z;
    struct vec3 best_normal_return;
    bool best_outside;
    bool exists = false;
    struct vec3 color;
    struct material material;
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
        double vec_check = vec_dot(vec_sub(ray.p1,ray.p2), outward_normal);
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
            material = spheres[i].material;
        }
    }
    return (struct normal_ray) { .sphere_point = {best_x,best_y,best_z}, .normal = best_normal_return, .t = closest_t, .outside_face = best_outside, .exists = exists, .material = material };
}

struct vec3 back_map(double px, double py, struct vec3 pos, struct vec3 up, struct vec3 right) {
    // assume viewport centered at origin and in xy plane
    // start at the origin
    double x = -1.0 * ((px/(double)IMAGE_WIDTH * (double) VIEWPORT_WIDTH) - VIEWPORT_WIDTH/2.0);
    double y = -1.0 * ((py/(double) IMAGE_HEIGHT * (double) VIEWPORT_HEIGHT) - VIEWPORT_HEIGHT/2.0);
    double z = 0.0;
    struct vec3 right_thingy = vec_mul_scalar(x, right);
    struct vec3 up_thingy = vec_mul_scalar(y, up);
    return vec_add(
        pos,
        vec_add(right_thingy, up_thingy)
    ); //vec add scalar {0.0,0.0,z}
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
    if (depth > 50) {
        return (struct vec3) { 0.0, 0.0, 0.0 };
    }
    struct sphere spheres[] = {
        {.center = {0.0, -100.5, -1.0}, .radius = 100.0, .material = {.tag = DIFFUSE, .data.diffuse = {0.8,0.8,0.0}}}, //moved by 5
        {.center = {0.0, 0.0,-1.2}, .radius = 0.5, .material = {.tag = DIFFUSE, .data.diffuse = {0.1, 0.2, 0.5}}},
        {.center = {-1.0, 0.0, -1.0}, .radius = 0.5, .material = {.tag = DIELECTRIC, .data.dielectric = {.refraction_index = 1.5}}},
        {.center = {-1.0, 0.0, -1.0}, .radius = 0.4, .material = {.tag = DIELECTRIC, .data.dielectric = {.refraction_index = 1.0 / 1.5}}},
        {.center = {1.0, 0.0, -1.0}, .radius = 0.5, .material = {.tag = METAL, .data.metal = {.albedo = {0.8,0.6,0.2}, .fuzz = 1.0}}}
    };

     // {.albedo = {0.2,0.2,0.2}, .fuzz = 0.0}}
    struct normal_ray normal_from_point = intersect_t(spheres, sizeof(spheres)/sizeof(struct sphere), ray);
    if (normal_from_point.exists) {
        if (normal_from_point.material.tag == DIFFUSE) {
            struct vec3 center_of_reflection = vec_add(normal_from_point.normal, normal_from_point.sphere_point);
            struct vec3 reflection_offset = vec_add(center_of_reflection, random_sphere_generator());
            struct ray reflected_ray = {normal_from_point.sphere_point, reflection_offset};
            return vec_mul(normal_from_point.material.data.diffuse.color, ray_color(reflected_ray, depth+1));
        } else if (normal_from_point.material.tag == METAL) {
            struct vec3 reflected_ray = vec_add(
                normalize(
                    reflect(ray_to_vec(ray), normal_from_point.normal)
                ),
                vec_mul_scalar(normal_from_point.material.data.metal.fuzz, random_sphere_generator())
            );

            if (vec_dot(reflected_ray, normal_from_point.normal) < 0.0) {
                return (struct vec3) {0.0,0.0,0.0};
            }
            struct vec3 refl_color = ray_color(
                (struct ray) {
                    normal_from_point.sphere_point,
                    vec_add(reflected_ray, normal_from_point.sphere_point)
                },
                depth+1
            );
            return vec_mul(normal_from_point.material.data.metal.albedo, refl_color);
        } else if (normal_from_point.material.tag == DIELECTRIC) {
            double index;
            if (normal_from_point.outside_face) {
                index = 1.0/normal_from_point.material.data.dielectric.refraction_index;
            } else {
                index = normal_from_point.material.data.dielectric.refraction_index;
            }
            struct vec3 unit_direction = normalize(ray_to_vec(ray));
            double cos_theta = fmin(vec_dot(vec_mul_scalar(-1.0, unit_direction), normal_from_point.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            struct vec3 refract_direction;
            if (index * sin_theta > 1.0 || reflectance(cos_theta, index) > ((double) rand()/RAND_MAX)) {
                refract_direction = reflect(unit_direction, normal_from_point.normal);
            } else {
                refract_direction = refract(unit_direction,normal_from_point.normal,index);
            }
            struct ray out_ray = {
                normal_from_point.sphere_point,
                vec_add(normal_from_point.sphere_point, refract_direction)
            };
            return ray_color(out_ray, depth+1);
        } else {
            fprintf(stderr, ":(");
            exit(1);
        }
    } else {
        return (struct vec3) { 0.61176470588, 0.76470588235, 0.90196078431 }; // Background color!
        // return (struct vec3) { 0.8, 0.8, 1.0 };
    }
}


struct vec3 pixel_color(int px, int py, int sample_size) { //make use sample sizes
    // (30, 50) (29.5..30.5, 49.5..50.5)
    // randomize px and py
    // rand() = random int between 0 and rand_max
    // (rand()/RAND_MAX)-0.5
    //
    double camera_x = 0.0;
    double camera_y = 1.5;
    double camera_z = -1.0;
    double focal_distance = 0.5;
    struct vec3 focal_point = {camera_x, camera_y, camera_z};
    struct vec3 sum = {0.0,0.0,0.0};
    struct vec3 back_map_pos = {camera_x, camera_y + focal_distance, camera_z};
    for (int i = 0; i < sample_size; i++) {
        double anti_aliasing_offset = ((double) rand()/RAND_MAX) - 0.5;
        struct vec3 viewport_location = back_map(px + anti_aliasing_offset, py + anti_aliasing_offset,
            back_map_pos,
            (struct vec3) { 0.0, 0.0, 1.0 },
            (struct vec3) { 1.0, 0.0, 0.0 }
        );
        struct ray ray = {focal_point, vec_add(focal_point, vec_sub(focal_point, viewport_location))};
        sum = vec_add(sum, ray_color(ray, 0));
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
