// Define constants for various parameters used in the shader
#define MAX_DIST 100.0 // Maximum ray march distance
#define MAX_STEPS 128 // Maximum number of steps for ray marching
#define SURF_DIST 0.01 // Surface distance threshold for hit detection
#define M_PI 3.1415926535897932384626433832795 // Pi constant for mathematical calculations
#define AA 1 // Anti-aliasing setting


// Material definitions for different parts of the scene
#define MATERIAL_BODY 1
#define MATERIAL_BELLY 2
#define MATERIAL_SKIN_YELLOW 3
#define MATERIAL_EYE 4
#define MATERIAL_OCEAN 5
#define MATERIAL_GLACIER 6 // New material for the glacier
#define MATERIAL_HEART_EYE 7


// Fresnel effect and water specular highlight strength settings
#define FRESNEL_POWER 5.0 // Controls the intensity of the Fresnel effect
#define WATER_SPECULAR_STRENGTH .9// Strength of specular highlights on the water


// Macro to update the closest hit information
#define check_hit(m, d) if(d <= mindist) { material = m; mindist = d; }


// Structure to represent a hit in the ray marching algorithm
struct Hit {
    float d;
    int material;
};


// Ocean wave parameters for generating procedural ocean waves
const float waveAmp = 0.3; // Wave amplitude
const float waveLength = 2.0; // Wave length
const float waveSpeed = 2.0; // Wave speed
const int waveOctaves = 5; // Number of wave octaves for fractal sum
const float waveFalloff = 0.5; // Falloff for each octave


// Hash function for generating random values based on input
float hash(float n) { return fract(sin(n) * 43758.5453123); }


// Noise function for generating procedural textures
float noise(vec3 x) {
    // Split the input vector into integer and fractional parts
    vec3 p = floor(x);
    vec3 f = fract(x);
    // Smoothen the fractional part
    f = f * f * (3.0 - 2.0 * f);


    // Generate a unique number based on the integer part
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    // Interpolate hashed values based on the fractional part to get the noise value
    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                   mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                   mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}


// Fractal Brownian Motion function for adding detail to textures
float fbm(vec3 x) {
    float v = 0.0;
    float a = 0.5;
    float shift = 10.0;
    // Loop to accumulate noise at different frequencies and amplitudes
    for (int i = 0; i < 4; ++i) { // Adjust octaves for more/less detail
        v += a * noise(x);
        x = x * 2.0 + shift; // Scale and shift the input for the next octave
        a *= 0.5; // Reduce amplitude for higher octaves
    }
    return v;
}


// Signed distance functions (SDF) for different geometric primitives and transformations


float sdOcean(vec3 p) {
    // Calculate ocean wave height using fbm and wave parameters
    float height = fbm(p * vec3(1.0/waveLength, 1, 1.0/waveLength) + vec3(0, waveSpeed*iTime, 0));
    // Return the signed distance to the ocean surface
    return p.y - (waveAmp * height);
}


float sd_sphere(vec3 p, float r) {
    // SDF for a sphere centered at the origin with radius r
    return length(p) - r;
}


float sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
    // SDF for a capsule defined by end points a and b and radius r
    vec3 ap = p - a;
    vec3 ab = b - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(ap - ab*t) - r;
}


float sd_plane(vec3 p, vec3 n) {
    // SDF for a plane with normal vector n
    return dot(p, n);
}


float sd_torus(vec3 p, vec2 r) {
    // SDF for a torus with major radius r.x and minor radius r.y
    float x = length(p.xz) - r.x;
    return length(vec2(x, p.y)) - r.y;
}


float sd_ellipsoid(vec3 p, vec3 r) {
    // SDF for an ellipsoid with radius vector r
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}


float sd_round_cone(vec3 p, float r1, float r2, float h) {
    // SDF for a rounded cone with radii r1, r2 and height h
    vec2 q = vec2(length(p.xz), p.y);
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
    return dot(q, vec2(a,b) ) - r1;
}


float sd_glacier_square(vec3 p, vec3 size) {
    // Basic box SDF for the glacier with added noise for ice-like features
    vec3 d = abs(p) - size;
    float basicShape = min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
    float noiseStrength = 0.25*pow((iTime/60.0) + 1.0, 0.25); // Adjust the strength of the noise
    float noiseScale = 2.0; // Scale of the noise
    float detail = fbm(p * noiseScale) * noiseStrength;
    return basicShape + detail - 0.1; // 0.1 is the smoothing radius
}
float sd_glacier_circle(vec3 p, float r, float noiseStrength) {
    // Basic box SDF for the glacier
    float d_from_circ = sqrt(p.x * p.x + p.z * p.z) - r;
    float vertical_d = abs(p.y) - 0.5;
    float basicShape = 0.0;
    if (vertical_d <= 0.0 && d_from_circ <= 0.0) {
        basicShape = 0.0;
    }
    else if (vertical_d <= 0.0) {
        basicShape = d_from_circ;
    }
    else if (d_from_circ <= 0.0) {
        basicShape = vertical_d;
    }
    else {
        basicShape = sqrt(vertical_d * vertical_d + d_from_circ * d_from_circ);
    }
    // Adding noise for ice-like features
    float noiseScale = 5.0; // Scale of the noise
    float detail = fbm(p * noiseScale) * noiseStrength;


    return basicShape + detail - 0.1; // 0.1 is the smoothing radius
}


float edgeDistance(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}


float hexagonSDF(vec3 r, float glacierSize, float noiseStrength) {
    vec2 p = vec2(r.x, r.z);
    vec2 hexagonPoints[6] = vec2[](vec2(-3, 5) * glacierSize, vec2(3, 6) * glacierSize, vec2(7, 2) * glacierSize, vec2(5, -3) * glacierSize, vec2(-4, -5) * glacierSize, vec2(-6, -1) * glacierSize);
    float distance = edgeDistance(p, hexagonPoints[0], hexagonPoints[1]);
    
    for (int i = 1; i < 6; ++i) {
        distance = min(distance, edgeDistance(p, hexagonPoints[i], hexagonPoints[(i + 1) % 6]));
    }
    float vertical_d = abs(r.y) - 1.0;
    if (vertical_d > 0.0) {
        distance = sqrt(distance * distance + vertical_d * vertical_d);
    }
    
    // Check if point is inside the hexagon to return negative distance
    // This is a simplified inside-outside check and may need refinement
    bool inside = true;
    for (int i = 0; i < 6; ++i) {
        vec2 a = hexagonPoints[i];
        vec2 b = hexagonPoints[(i + 1) % 6];
        vec2 toPoint = p - a;
        vec2 edgeDir = b - a;
        if ((edgeDir.x * toPoint.y - edgeDir.y * toPoint.x) > 0.0) {
            inside = false;
            break;
        }
    }


    if (inside) {
        if (vertical_d > 0.0) {
            distance = vertical_d;
        }
        else {
            distance = 0.0;
        }
    }
    float noiseScale = 3.0; // Scale of the noise
    float detail = fbm(r * noiseScale) * noiseStrength;
    
    return distance + detail - 0.1;
}


float op_smooth_union(float d1, float d2, float k ) {
    float h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0);
    return mix( d2, d1, h ) - k*h*(1.0-h);
}


vec3 rotate_x(vec3 p, float t) {
    float cost = cos(t);
    float sint = sin(t);
    return mat3(1.0, 0.0, 0.0, 0.0, cost, -sint, 0.0, sint, cost) * p;
}


vec3 rotate_z(vec3 p, float t) {
    float cost = cos(t);
    float sint = sin(t);
    return mat3(cost, -sint, 0.0, sint, cost, 0.0, 0.0, 0.0, 1.0) * p;
}


void get_penguin_sdf(vec3 p, inout float mindist, inout int material) {
    bool happy = 2.0 * sin(iTime/15.0) + 2.7 >= 1.5;
    float dist = op_smooth_union(sd_ellipsoid(p - vec3(0.0, 1.9, 0.0), vec3(2.5, 5.2, 2.0)), sd_sphere(p - vec3(0.0, 6.5, 0.0), 1.8), 0.5);
    const vec3 larm_pos = vec3(-2.0, 4.0, 0);
    const vec3 rarm_pos = larm_pos*vec3(-1.0, 1.0, 1.0);
    const vec3 arm_radius = vec3(0.4, 1.9, 0.5);
    const vec3 tail_pos = vec3(-0, -0.5, -1.9);
    const vec3 tail_radius = vec3(1.0, 0.4, 1.2);


    if (happy) {
        dist = op_smooth_union(dist, sd_ellipsoid(rotate_z(p - larm_pos, -0.7 - sin(iTime*5.0)*0.2), arm_radius), 0.1);
        dist = op_smooth_union(dist, sd_ellipsoid(rotate_z(p - rarm_pos, 0.7 - sin(iTime*5.0)*0.2), arm_radius), 0.1);
    }
    else {
        const vec3 larm_pos = vec3(-2.0, 5.0, 0);
        const vec3 rarm_pos = larm_pos*vec3(-1.0, 1.0, 1.0);
        dist = op_smooth_union(dist, sd_ellipsoid(rotate_z(p - larm_pos, 0.7 + sin(iTime*10.0)*0.2), arm_radius), 0.1);
        dist = op_smooth_union(dist, sd_ellipsoid(rotate_z(p - rarm_pos,-0.7 + sin(iTime*10.0)*0.2), arm_radius), 0.1);
    }


    
    dist = op_smooth_union(dist, sd_ellipsoid(p - tail_pos, tail_radius), 0.9);
    const vec3 leye_pos = vec3(-0.5, 6.8, 1.2);
    const vec3 reye_pos = leye_pos*vec3(-1.0, 1.0, 1.0);
    dist = op_smooth_union(dist, sd_torus(rotate_x(p - leye_pos, M_PI/2.0), vec2(0.8, 0.2)), 0.1);
    dist = op_smooth_union(dist, sd_torus(rotate_x(p - reye_pos, M_PI/2.0), vec2(0.8, 0.2)), 0.1);
    check_hit(MATERIAL_BODY, dist);
    dist = sd_ellipsoid(p - vec3(0.0, 1.8, 0.1), vec3(2.6, 5.3, 2.3)*0.9);
    check_hit(MATERIAL_BELLY, dist);
    const vec3 lfoot_pos = vec3(-1.2, -0.5, 1.9);
    const vec3 rfoot_pos = vec3(-lfoot_pos.x, lfoot_pos.y, lfoot_pos.z);
    const vec3 foot_radius = vec3(1.0, 0.3, 1.0);
    if (happy) {
        dist = min(sd_ellipsoid(rotate_z(p - lfoot_pos, sin(iTime*5.0)*0.3), foot_radius), sd_ellipsoid(rotate_z(p - rfoot_pos, sin(iTime*5.0+M_PI)*0.3), foot_radius));
    }
    else {
        dist = min(sd_ellipsoid(rotate_x(p - lfoot_pos, sin(iTime*10.0)*0.3), foot_radius), sd_ellipsoid(rotate_x(p - rfoot_pos, sin(iTime*10.0+M_PI)*0.3), foot_radius));
    }
    dist = min(dist, sd_round_cone(rotate_x(p - vec3(0.0, 5.5, 2.0), 3.3*M_PI/2.0), 0.1, 0.4, 0.6));
    check_hit(MATERIAL_SKIN_YELLOW, dist);
    dist = min(sd_ellipsoid(p - leye_pos, vec3(0.8, .9, 0.7)), sd_ellipsoid(p - reye_pos, vec3(0.8, .9, 0.7)));
    check_hit(MATERIAL_EYE, dist);
    vec3 lretina_pos = vec3(-0.4 + sin(iTime)*-0.2, 6.8,1.9);
    vec3 rretina_pos = vec3(0.45 - sin(iTime)*-0.2, 6.8, 1.9);
    if (happy) {
        dist = min(op_smooth_union(sd_ellipsoid(rotate_z(p - lretina_pos + vec3(0.2, 0, 0), 0.5), vec3(0.2, 0.3, 0.1)), sd_ellipsoid(rotate_z(p - lretina_pos, -0.5), vec3(0.2, 0.3, 0.1)), 0.06), op_smooth_union(sd_ellipsoid(rotate_z(p - rretina_pos, 0.5), vec3(0.2, 0.3, 0.1)), sd_ellipsoid(rotate_z(p - rretina_pos - vec3(0.2, 0, 0), -0.5), vec3(0.2, 0.3, 0.1)), 0.06));
        check_hit(MATERIAL_HEART_EYE, dist);
    }
    else {
        vec3 lretina_pos = vec3(-0.4 + sin(iTime)*-0.2, 6.8,1.9);
        vec3 rretina_pos = vec3(0.4 - sin(iTime)*-0.2, 6.8, 1.9);
        dist = min(sd_ellipsoid(p - lretina_pos, vec3(0.2, 0.3, 0.1)), sd_ellipsoid(p - rretina_pos, vec3(0.2, 0.25, 0.1)));
        check_hit(MATERIAL_BODY, dist);
    }
}


Hit get_scene_sdf(vec3 p) {
    float mindist = MAX_DIST;
    int material = 0;


    get_penguin_sdf(p, mindist, material);


    // Glacier integration
    vec3 glacierPos = vec3(0.0, -1.5, 0.0); // Position of the glacier
    float glacierSize = 2.0 * sin(iTime/15.0) + 2.7;


    float distGlacier = hexagonSDF(p - glacierPos, glacierSize, 0.15);
    check_hit(MATERIAL_GLACIER, distGlacier);


    float distOcean = sdOcean(p - vec3(0.0, -1.0, 0.0));
    check_hit(MATERIAL_OCEAN, distOcean);


    return Hit(mindist, material);
}
vec3 get_normal(vec3 p) {
    const float eps = 0.001; // Increase epsilon for normal calculation
    vec2 e = vec2(1.0,-1.0);
    return normalize(e.xyy*get_scene_sdf(p + e.xyy*eps).d + e.yyx*get_scene_sdf(p + e.yyx*eps).d + e.yxy*get_scene_sdf(p + e.yxy*eps).d + e.xxx*get_scene_sdf(p + e.xxx*eps).d);
}


Hit ray_march(vec3 rayfrom, vec3 raydir) {
    float t = 0.0;
    Hit hit;
    for(int i=0; i<MAX_STEPS; ++i) {
        vec3 p = rayfrom + t*raydir;
        hit = get_scene_sdf(p);
        t += hit.d;
        if(hit.d <= SURF_DIST || t >= MAX_DIST)
            break;
    }
    hit.d = t;
    return hit;
}


float hard_shadow(vec3 rayfrom, vec3 raydir, float tmin, float tmax) {
    float bias = 0.02; // Small bias to offset the ray start position
    float t = tmin + bias; // Apply bias to the initial t value
    for(int i=0; i<MAX_STEPS; i++) {
        vec3 p = rayfrom + raydir*t;
        float h = get_scene_sdf(p).d;
        if(h < SURF_DIST) return 0.0; // If hit distance is below threshold, in shadow
        t += h;
        if(t > tmax) break;
    }
    return 1.0; // No obstruction found, not in shadow
}




float get_occlusion(vec3 rayfrom, vec3 normal) {
    const int AO_ITERATIONS = 5;
    const float AO_START = 0.02; // Increase start distance to reduce self-shadowing artifacts
    const float AO_DELTA = 0.1;
    const float AO_DECAY = 0.95;
    const float AO_INTENSITY = 2.0;
    float occ = 0.0;
    float decay = 1.0;
    for(int i=0; i<AO_ITERATIONS; ++i) {
        float h = AO_START + float(i) * AO_DELTA;
        float d = get_scene_sdf(rayfrom + normal*h).d;
        occ += (h-d) * decay;
        decay *= AO_DECAY;
    }
    return clamp(1.0 - occ * AO_INTENSITY, 0.0, 1.0);
}




vec3 get_material_diffuse(vec3 p, int material) {
    switch(material) {
        case MATERIAL_BODY:
            return vec3(0.0, 0.0, 0.0);
        case MATERIAL_BELLY:
            return vec3(0.6, 0.6, 0.6);
        case MATERIAL_OCEAN:
            float pattern = sin(p.x * 0.2 + iTime) * 0.05 + sin(p.z * 0.2 + iTime * 1.5) * 0.05;
            return vec3(0.1, 0.3, 0.5 + pattern);
        case MATERIAL_SKIN_YELLOW:
            return vec3(1.0, .3, .01);
        case MATERIAL_EYE:
            return vec3(1.0, 1.0, 1.0);
        case MATERIAL_HEART_EYE:
            return vec3(1.0, 0.0, 0.0);
        default:
            return vec3(1.0, 1.0, 1.0);
    }
}




vec3 get_material_specular(vec3 p, int material) {
    switch(material) {
        case MATERIAL_BODY:
            return vec3(0.6, 0.6, 0.6);
        case MATERIAL_SKIN_YELLOW:
            return vec3(1.0, .6, .1);
        case MATERIAL_EYE:
            return vec3(1.0, 1.0, 1.0);
        case MATERIAL_OCEAN:
            // Increase specular strength for water
            return vec3(WATER_SPECULAR_STRENGTH);
        default:
            return vec3(0.0, 0.0, 0.0);
    }
}




vec3 get_light(vec3 raydir, vec3 p, int material) {
    vec3 diffuse = vec3(0);
    vec3 specular = vec3(0);
    vec3 normal = get_normal(p);
    float occlusion = get_occlusion(p, normal);
    const float SUN_INTENSITY = 1.5;
    const float SUN_SHINESS = 10.0;
    const vec3 SUN_POS = vec3(-10.0, 20.0, 10.0);
    const vec3 SUN_COLOR = vec3(1.0,0.77,0.6);
    vec3 sun_vec = SUN_POS - p;
    vec3 sun_dir = normalize(sun_vec);
    float sun_diffuse = clamp(dot(normal, sun_dir), 0.0, 1.0);
    float sun_shadow = hard_shadow(p, sun_dir, 0.1, length(sun_vec));
    float sun_specular = pow(clamp(dot(reflect(sun_dir, normal), raydir), 0.0, 1.0), SUN_SHINESS);
    diffuse += SUN_COLOR * (sun_diffuse * sun_shadow * SUN_INTENSITY);
    specular += SUN_COLOR * sun_specular;
    const float SKY_INTENSITY = 0.3;
    const float SKY_SHINESS = 10.0;
    const float SKY_MINIMUM_ATTENUATION = 0.0;
    const vec3 SKY_COLOR = vec3(0.5, 0.7, 1.0);
    float sky_diffuse = SKY_MINIMUM_ATTENUATION + SKY_MINIMUM_ATTENUATION * normal.y;
    float sky_specular = pow(clamp(dot(reflect(vec3(0.0,1.0,0.0), normal), raydir), 0.0, 1.0), SKY_SHINESS);
    diffuse += SKY_COLOR * (SKY_INTENSITY * sky_diffuse * occlusion);
    specular += SKY_COLOR * (sky_specular * occlusion);
    const float INDIRECT_INTENSITY = 0.2;
    const float INDIRECT_SHINESS = 10.0;
    const vec3 INDIRECT_COLOR = SUN_COLOR;
    vec3 ind_dir = normalize(sun_dir * vec3(-1.0,0.0,-1.0));
    float ind_diffuse = clamp(dot(normal, ind_dir), 0.0, 1.0);
    float ind_specular = pow(clamp(dot(reflect(ind_dir, normal), raydir), 0.0, 1.0), INDIRECT_SHINESS);
    diffuse += INDIRECT_COLOR * (ind_diffuse * INDIRECT_INTENSITY);
    specular += INDIRECT_COLOR * (ind_specular * INDIRECT_INTENSITY);
    const vec3 ENV_COLOR = SKY_COLOR;
    const float ENV_INTENSITY = 0.3;
    diffuse += ENV_COLOR * ENV_INTENSITY;
    if(material == MATERIAL_OCEAN) {
        float fresnel = pow(1.0 - max(dot(normal, -raydir), 0.0), FRESNEL_POWER);
        specular += fresnel * vec3(WATER_SPECULAR_STRENGTH);
    }
    vec3 col = diffuse * get_material_diffuse(p, material) + specular * get_material_specular(p, material);


    col = pow(col, vec3(0.4545));
    return col;
}


vec3 get_ray(vec3 lookfrom, vec3 lookat, float tilt, float vfov, vec2 uv) {
    vec3 vup = vec3(sin(tilt), cos(tilt), 0.0);
    vec3 lookdir = normalize(lookat - lookfrom);
    vec3 u = cross(lookdir, vup);
    vec3 v = cross(u, lookdir);
    vec3 w = lookdir * (1.0 / tan(vfov*M_PI/360.0));
    mat3 t = mat3(u, v, w);
    return normalize(t * vec3(uv, 1.0));
}


vec3 render(vec2 uv) {
    float time  = iTime*0.1+1.4;
    vec2  mouse = iMouse.xy/iResolution.xy;
    vec3 lookfrom = vec3(cos(time+mouse.x*6.28),0.3,sin(time+mouse.x*6.28))*50.0;
    vec3 lookat = vec3(0, 1, 0);
    lookfrom += lookat+vec3(0,15,0);
    vec3 raydir = get_ray(lookfrom, lookat, 0.0, 20.0, uv);
    Hit hit = ray_march(lookfrom, raydir);
    vec3 p = lookfrom + raydir * hit.d;
    return get_light(raydir, p, hit.material);
}
vec3 render_aa(vec2 uv) {
#if AA > 1
    float w = 1.0/iResolution.y;
    vec3 col = vec3(0.0);
    for(int n=0; n<AA*AA; ++n) {
        vec2 o = 2.0*(vec2(float(int(n / AA)),float(int(n % AA))) / float(AA) - 0.5);
        col += render(uv + o*w);
    }
    col /= float(AA*AA);
    return col;
#else
    return render(uv);
#endif
}


void mainImage(out vec4 fragcolor, in vec2 fragcoord) {
    vec2 uv = 2.0 * ((fragcoord-0.5*iResolution.xy) / iResolution.y);
    vec3 col = render_aa(uv);
    fragcolor = vec4(col,1);
}







