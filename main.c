#include <stdio.h>
#include <iso646.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <tgmath.h>

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 600
#define SAMPLE_SIZE 10

#define VIEWPORT_WIDTH 2.0
#define VIEWPORT_HEIGHT ((VIEWPORT_WIDTH * IMAGE_HEIGHT)/IMAGE_WIDTH)

#define cahr char

double rand_double(double low, double high) {
    double zero_to_one = (double)rand() / (double)RAND_MAX;
    return low + (high-low)*zero_to_one;
}

typedef struct vec3 {double x, y, z;} vec3;

typedef struct ray {vec3 p1, p2; double time;} ray;

typedef struct Dielectric {double refraction_index;} Dielectric;

typedef struct Metal {vec3 albedo; double fuzz;} Metal;

typedef struct Diffuse {vec3 color;} Diffuse;

// Either:
// - Diffuse, with vec3 color
// Or:
// - Metal, with vec3 albedo and double fuzz
// Or:
// - Dielectic, with double refraction_index

typedef union materialUnion {
    Diffuse diffuse;
    Metal metal;
    Dielectric dielectric;
} materialUnion;

typedef enum materialTag {
    DIFFUSE,
    METAL,
    DIELECTRIC,
} materialTag;

typedef struct material {
    materialUnion data;
    materialTag tag;
} material;

typedef struct sphere {ray center; double radius; material material;} sphere;

typedef struct bbox {vec3 begin, end;} bbox;

typedef struct bbox_tree_internal {
    struct bbox_tree *child_1;
    struct bbox_tree *child_2;
} bbox_tree_internal;

typedef struct bbox_tree_leaf {
    sphere sphere_1;
    sphere sphere_2;
    int n_spheres;
} bbox_tree_leaf;

typedef union bbox_tree_union {
    bbox_tree_internal intenal;
    bbox_tree_leaf leaf;
} bbox_tree_union;

typedef struct bbox_tree {
    bbox box;
    bbox_tree_union tree;
    bool is_leaf;
} bbox_tree;

double vec_dist(vec3 p1, vec3 p2) {
    return sqrt(((p2.x-p1.x) * (p2.x-p1.x)) + ((p2.y-p1.y) * (p2.y-p1.y)) + ((p2.z-p1.z) * (p2.z-p1.z)));
}

double vec_len(vec3 p1) {
    return sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
}

double vec_dot(vec3 v1, vec3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

vec3 vec_cross(vec3 p1, vec3 p2) {
    return (vec3) {p1.y * p2.z - p1.z * p2.y, p1.z * p2.x - p1.x * p2.z, p1.x * p2.y - p1.y * p2.x};
}

vec3 vec_mul(vec3 v1, vec3 v2) {
    return (vec3) {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}

vec3 vec_mul_scalar(double scalar, vec3 vec) {
    return (vec3) { vec.x*scalar, vec.y*scalar, vec.z*scalar };
}

vec3 vec_div(vec3 v1, vec3 v2) {
    return (vec3) {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};
}

vec3 vec_div_scalar(vec3 vec, double scalar) {
    return (vec3) { vec.x/scalar, vec.y/scalar, vec.z/scalar };
}

vec3 vec_add(vec3 v1, vec3 v2) {
    return (vec3) {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

vec3 normalize(vec3 vec) {
    return vec_div_scalar(vec, vec_len(vec));
}

vec3 ray_to_vec(ray ray) {
    return (vec3) { ray.p2.x - ray.p1.x, ray.p2.y - ray.p1.y, ray.p2.z - ray.p1.z };
}

double sq(double x) {
    return x * x;
}

double avg(double x, double y) {
    return (x + y)/2.0;
}

vec3 vec_sub(vec3 p1, vec3 p2) {
    return (vec3) {(p1.x - p2.x), (p1.y - p2.y), (p1.z - p2.z)};
}

vec3 map_to_time(double time, ray ray) {
    double x_dist = ray.p2.x-ray.p1.x;
    double y_dist = ray.p2.y-ray.p1.y;
    double z_dist = ray.p2.z-ray.p1.z;
    vec3 mapped_point = vec_add((vec3) {x_dist * time, y_dist * time, z_dist * time}, ray.p1);
    return mapped_point;
}

typedef struct normal_ray {vec3 sphere_point; vec3 normal; double t; bool outside_face; bool exists; material material;} hit_record;

bbox bound_spheres(sphere *spheres, int length, double t) {
    vec3 min_length = {INFINITY, INFINITY, INFINITY};
    for (int i = 0; i < length; i++) {
        vec3 sphere_pos = map_to_time(t, spheres[i].center);
        if (sphere_pos.x + spheres[i].radius < min_length.x) {
            min_length.x = sphere_pos.x;
        } else if (sphere_pos.y + spheres[i].radius < min_length.y) {
            min_length.y = sphere_pos.y;
        } else if (sphere_pos.z + spheres[i].radius < min_length.z) {
            min_length.z = sphere_pos.z;
        }
    }

    vec3 max_length = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < length; i++) {
        vec3 sphere_pos = map_to_time(t, spheres[i].center);
        if (sphere_pos.x + spheres[i].radius > max_length.x) {
            max_length.x = sphere_pos.x;
        } else if (sphere_pos.y + spheres[i].radius > max_length.y) {
            max_length.y = sphere_pos.y;
        } else if (sphere_pos.z + spheres[i].radius > max_length.z) {
            max_length.z = sphere_pos.z;
        }
    }
    return (bbox) {.begin = min_length, .end = max_length};
}

typedef struct bbox_pair {
    bbox first;
    bbox second;
} bbox_pair;

bbox_pair bbox_x_split(bbox box) {
    return (bbox_pair) {
        .first = {.begin = box.begin, .end = {avg(box.begin.x, box.end.x), box.end.y,box.end.z}},
        .second = {.begin = {avg(box.begin.x, box.end.x), box.begin.y, box.begin.z}, .end = box.end}
    };
}

bbox_pair bbox_y_split(bbox box) {
    return (bbox_pair) {
        .first = {.begin = box.begin, .end = {box.end.x, avg(box.begin.y, box.end.y),box.end.z}},
        .second = {.begin = {box.begin.x, avg(box.begin.y, box.end.y), box.begin.z}, .end = box.end}
    };
}

bbox_pair bbox_z_split(bbox box) {
    return (bbox_pair) {
        .first = {.begin = box.begin, .end = {box.end.x, box.end.y, avg(box.begin.z, box.end.z)}},
        .second = {.begin = {box.begin.x, box.begin.y, avg(box.begin.z, box.end.z)}, .end = box.end}
    };
}

bbox_tree beatbox(sphere *spheres, int length, double t) {
    double choice = (double) rand()/RAND_MAX;
    bbox spheres_bound = bound_spheres(spheres, length, t);
    if (length > 2) {
        // do splitting
    } else {
        // return a single leaf node with that box
    }
    if (choice <= 0.33) {

    } else if (0.33 < choice && choice <= 0.66) {
        // y
    } else {
        // z
    }

}


vec3 refract(vec3 ray_in, vec3 normal, double etai_over_etat) {
    double cos_theta = fmin(vec_dot(vec_mul_scalar(-1.0,ray_in), normal), 1.0);
    vec3 r_out_perp = vec_mul_scalar(etai_over_etat, vec_add(ray_in, vec_mul_scalar(cos_theta, normal)));
    vec3 r_out_parallel = vec_mul_scalar(-1.0, vec_mul_scalar(sqrt(fabs(1.0 - sq(vec_len(r_out_perp)))), normal));
    return vec_add(r_out_perp, r_out_parallel);
}

double x_func(double x, ray trace) {
    return (x-trace.p1.x)/(trace.p2.x-trace.p1.x);
}

double y_func(double y, ray trace) {
    return (y-trace.p1.y)/(trace.p2.y-trace.p1.y);
}

double z_func(double z, ray trace) {
    return (z-trace.p1.z)/(trace.p2.z-trace.p1.z);
}


bool check_overlap(double begin_x, double end_x, double begin_y, double end_y) {
    return (begin_y <= begin_x && begin_x <= end_y) || (begin_x <= begin_y && begin_y <= end_x) || (begin_x >= begin_y && end_x <= end_y);
}



bool intersect_box(bbox hitbox, ray trace) {
    double begin_x = x_func(hitbox.begin.x, trace);
    double end_x = x_func(hitbox.end.x, trace);
    double begin_y = y_func(hitbox.begin.y, trace);
    double end_y = y_func(hitbox.end.y, trace);
    double begin_z = z_func(hitbox.begin.z, trace);
    double end_z = z_func(hitbox.end.z, trace);

    return check_overlap(begin_x, end_x, begin_y, end_y) && (check_overlap(begin_x, end_x, begin_z, end_z) && (check_overlap(begin_z, end_z, begin_y, end_y)
        && check_overlap(0.0,1.0, begin_x, end_x) && check_overlap(0.0,1.0, begin_y, end_y) && check_overlap(0.0,1.0, begin_z, end_z)));
}

vec3 reflect(vec3 ray_in, vec3 normal) {
    return vec_sub(ray_in, vec_mul_scalar(2.0, vec_mul_scalar(vec_dot(ray_in, normal), normal)));
}

double reflectance(double cosine, double refraction_index) {
    double r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1-r0) * pow((1-cosine), 5);
}

ray still(double x, double y, double z) {
    return (ray) {{x,y,z}, {x,y,z}};
}

hit_record intersect_t(sphere *spheres, int length, ray ray) {
    bool exists = false;
    hit_record closest_hit = { .exists = false };
    // modify to work with sphere
    for (int i = 0; i < length; i++) {
        vec3 timed_center = map_to_time(ray.time, spheres[i].center);
        double t = -1.0;
        double a = vec_dot(ray.p2, ray.p2) + vec_dot(ray.p1, ray.p1) - 2*vec_dot(ray.p1, ray.p2);
        double b = 2 * vec_dot(vec_sub(ray.p1, timed_center), vec_sub(ray.p2, ray.p1));
        double c = -sq(spheres[i].radius) + vec_dot(ray.p1, ray.p1) + vec_dot(timed_center, timed_center) - 2 * vec_dot(ray.p1, timed_center);
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
        vec3 outward_normal = {
            (x - timed_center.x)/spheres[i].radius,
            (y - timed_center.y)/spheres[i].radius,
            (z - timed_center.z)/spheres[i].radius
        };

        vec3 normal_return;
        bool outside;
        double vec_check = vec_dot(vec_sub(ray.p1,ray.p2), outward_normal);
        if (vec_check >= 0.0) { // outside
            normal_return = outward_normal;
            outside = true;
        } else { // not outside
            normal_return = vec_mul((vec3) {-1.0,-1.0,-1.0}, outward_normal);
            outside = false;
        }
        if (!closest_hit.exists || closest_hit.t == -1.0 || (t != -1.0 && t < closest_hit.t)) {
            closest_hit = (hit_record) {
                .exists = true,
                .t = t, .sphere_point = { x, y, z }, .normal = normal_return, .outside_face = outside, .material = spheres[i].material
            };
        }
    }
    return closest_hit;
}

vec3 back_map(double px, double py, vec3 pos, vec3 up, vec3 right) {
    // assume viewport centered at origin and in xy plane
    // start at the origin
    double x = -1.0 * ((px/(double)IMAGE_WIDTH * (double) VIEWPORT_WIDTH) - VIEWPORT_WIDTH/2.0);
    double y = -1.0 * ((py/(double) IMAGE_HEIGHT * (double) VIEWPORT_HEIGHT) - VIEWPORT_HEIGHT/2.0);
    vec3 right_thingy = vec_mul_scalar(x, right);
    vec3 up_thingy = vec_mul_scalar(y, up);
    return vec_add(
        pos,
        vec_add(right_thingy, up_thingy)
    );
}

vec3 random_sphere_generator(void) { // TODO: optimize
    while (true) {
        double offset_x = rand_double(-1.0, 1.0);
        double offset_y = rand_double(-1.0, 1.0);
        double offset_z = rand_double(-1.0, 1.0);
        vec3 offset = {offset_x, offset_y, offset_z};
        double len = vec_len(offset);
        if (len < 1.0 && len > 1e-9) {
            return vec_div(offset,(vec3) {len,len,len});
        }
    }
}

vec3 random_disk_generator(double radius, vec3 right, vec3 up, vec3 disk_center) { // TODO: optimize
    while (true) {
        double offset_x = rand_double(-1.0, 1.0);
        double offset_y = rand_double(-1.0, 1.0);
        vec3 offset = {offset_x, offset_y, 0.0};
        double len = vec_len(offset);
        if (len < 1.0 && len > 1e-9) {
            vec3 norm_offset = normalize(offset);
            return vec_add(
                disk_center,
                vec_add(
                    vec_mul_scalar(radius, vec_mul_scalar(norm_offset.x, right)),
                    vec_mul_scalar(radius, vec_mul_scalar(norm_offset.y, up))
                )
            );
        }
    }
}

vec3 ray_color(ray trace, int depth, sphere *spheres, int n_spheres) {
    if (depth > 50) {
        return (vec3) { 0.0, 0.0, 0.0 };
    }
    hit_record normal_from_point = intersect_t(spheres, n_spheres, trace);
    if (normal_from_point.exists) {
        if (normal_from_point.material.tag == DIFFUSE) {
            vec3 center_of_reflection = vec_add(normal_from_point.normal, normal_from_point.sphere_point);
            vec3 reflection_offset = vec_add(center_of_reflection, random_sphere_generator());
            ray reflected_ray = {normal_from_point.sphere_point, reflection_offset, .time = trace.time};
            return vec_mul(normal_from_point.material.data.diffuse.color, ray_color(reflected_ray, depth+1, spheres, n_spheres));
        } else if (normal_from_point.material.tag == METAL) {
            vec3 reflected_ray = vec_add(
                normalize(
                    reflect(ray_to_vec(trace), normal_from_point.normal)
                ),
                vec_mul_scalar(normal_from_point.material.data.metal.fuzz, random_sphere_generator())
            );

            if (vec_dot(reflected_ray, normal_from_point.normal) < 0.0) {
                return (vec3) {0.0,0.0,0.0};
            }
            vec3 refl_color = ray_color(
                (ray) {
                    normal_from_point.sphere_point,
                    vec_add(reflected_ray, normal_from_point.sphere_point),
                    .time = trace.time
                },
                depth+1,
                spheres, n_spheres
            );
            return vec_mul(normal_from_point.material.data.metal.albedo, refl_color);
        } else if (normal_from_point.material.tag == DIELECTRIC) {
            double index;
            if (normal_from_point.outside_face) {
                index = 1.0/normal_from_point.material.data.dielectric.refraction_index;
            } else {
                index = normal_from_point.material.data.dielectric.refraction_index;
            }
            vec3 unit_direction = normalize(ray_to_vec(trace));
            double cos_theta = fmin(vec_dot(vec_mul_scalar(-1.0, unit_direction), normal_from_point.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            vec3 refract_direction;
            if (index * sin_theta > 1.0 || reflectance(cos_theta, index) > rand_double(0.0, 1.0)) {
                refract_direction = reflect(unit_direction, normal_from_point.normal);
            } else {
                refract_direction = refract(unit_direction,normal_from_point.normal,index);
            }
            ray out_ray = {
                normal_from_point.sphere_point,
                vec_add(normal_from_point.sphere_point, refract_direction),
                .time = trace.time
            };
            return ray_color(out_ray, depth+1, spheres, n_spheres);
        } else {
            fprintf(stderr, ":( %d", normal_from_point.material.tag);
            exit(1);
        }
    } else {
        return (vec3) { 0.61176470588, 0.76470588235, 0.90196078431 }; // Background color!
    }
}

vec3 pixel_color(int px, int py, int sample_size, sphere *spheres, int n_spheres) { //make use sample sizes
    vec3 look_from = {13.0, 2.0, 3.0};
    vec3 look_at = { 0.0, 0.0, 0.0};
    vec3 up = { 0.0, 1.0, 0.0 };
    double focal_length = 2.0; // TODO: Implement FOV instead of focal length
    double disk_radius = 0.001;

    vec3 forward = normalize(vec_sub(look_at, look_from));
    vec3 right = vec_cross(forward, up);
    vec3 disk_center = look_from;
    vec3 sum = {0.0,0.0,0.0};
    vec3 back_map_pos = vec_sub(disk_center, vec_mul_scalar(focal_length, forward));
    for (int i = 0; i < sample_size; i++) {
        double anti_aliasing_offset = rand_double(0.0, 1.0) - 0.5;
        vec3 viewport_location = back_map(
            px + anti_aliasing_offset, py + anti_aliasing_offset,
            back_map_pos,
            up,
            right
        );
        vec3 new_disk_focal = random_disk_generator(disk_radius, right, up, disk_center);
        ray ray = {new_disk_focal, vec_add(new_disk_focal, vec_sub(new_disk_focal, viewport_location)), .time = rand_double(0.0, 1.0)};
        sum = vec_add(sum, ray_color(ray, 0, spheres, n_spheres));
    }
    return vec_div(sum,(vec3) {(double) sample_size,(double) sample_size,(double) sample_size});
}



int main(void) {
    sphere spheres[488];
    material ground_material = { .tag = DIFFUSE, .data.diffuse = { .color = {0.5, 0.5, 0.5} } };
    spheres[0] = (sphere) { .center = still(0.0, -1000.0, 0.0), .radius = 1000.0, .material = ground_material };
    int index = 1;
    for (int i = -11; i < 11; i++) {
        for (int g = -11; g < 11; g++) {
            double choose_mat = rand_double(0.0, 1.0);
            vec3 center = {i + rand_double(0.0, 0.9), 0.2, g + rand_double(0.0, 0.9)};

            if (vec_len(vec_sub(center, (vec3) {4.0,0.2,0.0}))  > 0.9) {
                if (choose_mat < 0.8) {
                    // diffuse
                    vec3 color = {rand_double(0.0, 1.0),rand_double(0.0, 1.0),rand_double(0.0, 1.0)};
                    spheres[index] = (sphere) {
                        .center = {
                            .p1 = center,
                            .p2 = vec_add(
                                center,
                                (vec3) {0.0, rand_double(0.0, 0.5), 0.0}
                            )
                        },
                        .radius = 0.2,
                        .material = { .tag = DIFFUSE, .data.diffuse = { .color = color}}
                    };
                } else if (choose_mat < 0.95) {
                    // metal
                    vec3 albedo = vec_add(
                        (vec3) {0.5,0.5,0.5},
                        (vec3){ rand_double(0.0, 0.5), rand_double(0.0, 0.5), rand_double(0.0, 0.5) }
                    );
                    double fuzz = rand_double(0.0, 0.5);
                    spheres[index] = (sphere) {
                        .center = {.p1 = center, .p2 = center}, .radius = 0.2,
                        .material = { .tag = METAL, .data.metal = { .albedo = albedo, .fuzz = fuzz}}
                    };
                } else {
                    //glass
                    spheres[index] = (sphere) {
                        .center = {.p1 = center, .p2 = center}, .radius = 0.2,
                        .material = { .tag = DIELECTRIC, .data.dielectric = { .refraction_index = 1.5}}
                    };
                }
            }
            index++;
        }
    }
    spheres[485] = (sphere) {
        .center = still(0.0, 1.0, 0.0), .radius = 1.0,
        .material = { .tag = DIELECTRIC, .data.dielectric = { .refraction_index = 1.5 }}
    };
    spheres[486] = (sphere) {
        .center = still(-4.0, 1.0, 0.0), .radius = 1.0,
        .material = { .tag = DIFFUSE, .data.diffuse = { .color = {0.4, 0.2, 0.1} }}
    };
    spheres[487] = (sphere) {
        .center = still(4.0, 1.0, 0.0), .radius = 1.0,
        .material = { .tag = METAL, .data.metal = { .albedo = {0.7, 0.6, 0.5}, .fuzz = 0.0 } }
    };

    FILE *fptr = fopen("blah.ppm", "w");
    if (!fptr) {
        perror("Failed to open file");
        return 1;
    }
    // Write the PPM header
    fprintf(fptr, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    // Generate the gradient
    for (int py = IMAGE_HEIGHT-1; py >= 0; py--) {  // Iterate over height first
        printf("\r%lf%%", 100.0 - (100.0*(double)py/(double)IMAGE_HEIGHT));
        fflush(stdout);
        for (int px = 0; px < IMAGE_WIDTH; px++) {  // Iterate over width
            vec3 color = pixel_color(px, py, SAMPLE_SIZE, spheres, sizeof(spheres) / sizeof(struct sphere));
            fprintf(fptr, "%d %d %d ", (int)(sqrt(color.x)*255.0), (int)(sqrt(color.y)*255.0), (int)(sqrt(color.z)*255.0));
        }
        fprintf(fptr, "\n"); // Ensure each row starts on a new line
    }
    fclose(fptr);
    printf("\nPPM file generated: blah.ppm\n");
    return 0;
}
