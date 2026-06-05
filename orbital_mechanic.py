from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "Pygame is required to run The Orbital Mechanic.\n"
        "Install it with: python -m pip install -r requirements.txt"
    ) from exc


WIDTH, HEIGHT = 1180, 760
CENTER = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
BACKGROUND = (9, 12, 20)
PANEL = (21, 26, 39)
TEXT = (226, 232, 243)
MUTED = (142, 155, 181)
GRID = (32, 41, 58)
ORBIT_BLUE = (101, 190, 255)
MOON_GREY = (190, 198, 211)
EARTH_BLUE = (42, 126, 212)
EARTH_LAND = (64, 168, 91)
EARTH_ICE = (232, 244, 250)
SAT_GOLD = (255, 205, 87)
WARNING = (255, 112, 112)
GHOST = (190, 198, 211)
CRATER_DARK = (15, 9, 12)
CRATER_RIM = (95, 47, 40)
CRATER_GLOW = (255, 67, 37)
CRATER_CORE = (255, 180, 72)
FIRE_COLORS = [(255, 230, 123), (255, 159, 64), (255, 80, 52), (255, 45, 28)]
SMOKE_COLORS = [(90, 94, 103), (116, 119, 127), (148, 148, 151)]

G = 0.35
SOFTENING = 35.0
TIME_STEP = 0.18
MAX_TRAIL = 1500
LAUNCH_DISTANCE = 74.0
MOON_ORBIT_RADIUS = 245.0
PLANET_MASS = 22000.0
MOON_MASS = 190.0
SATELLITE_MASS = 2.5
DRAG_TO_VELOCITY = 0.035
PREDICTION_STEPS = 520
PREDICTION_SAMPLE_EVERY = 5
PREDICTION_MARGIN = 900
MOON_FLAME_DISTANCE = 150.0
MAX_PARTICLES = 950
TIME_SCALES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00]
NORMAL_TIME_INDEX = TIME_SCALES.index(1.00)
MIN_CAMERA_ZOOM = 0.25
MAX_CAMERA_ZOOM = 2.50
ZOOM_STEP = 1.12


@dataclass
class Body:
    name: str
    mass: float
    radius: int
    pos: pygame.Vector2
    vel: pygame.Vector2
    color: tuple[int, int, int]
    trail_color: tuple[int, int, int]
    gravity_scale: float = 1.0
    fixed: bool = False
    alive: bool = True
    trail: list[tuple[float, float]] = field(default_factory=list)

    def record_trail(self) -> None:
        self.trail.append((self.pos.x, self.pos.y))
        if len(self.trail) > MAX_TRAIL:
            self.trail.pop(0)


@dataclass
class Particle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    radius: float
    color: tuple[int, int, int]
    lifetime: float
    age: float = 0.0
    growth: float = 0.0
    drag: float = 0.98

    @property
    def alive(self) -> bool:
        return self.age < self.lifetime

    def update(self, dt: float) -> None:
        self.age += dt
        self.pos += self.vel * dt
        self.vel *= self.drag
        self.radius += self.growth * dt

    def draw(self, surface: pygame.Surface) -> None:
        fade = max(0.0, 1.0 - self.age / self.lifetime)
        alpha = int(230 * fade)
        size = max(2, int(self.radius * 2 + 4))
        particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size // 2, size // 2)
        pygame.draw.circle(
            particle_surface,
            (*self.color, alpha),
            center,
            max(1, int(self.radius)),
        )
        surface.blit(particle_surface, (self.pos.x - center[0], self.pos.y - center[1]))


@dataclass
class EarthCrater:
    normal: pygame.Vector2
    radius: float
    glow: float = 1.0


def acceleration(target: Body, bodies: list[Body]) -> pygame.Vector2:
    total = pygame.Vector2()
    for other in bodies:
        if other is target or not other.alive:
            continue
        delta = other.pos - target.pos
        dist_sq = delta.length_squared() + SOFTENING
        if dist_sq == 0:
            continue
        gravitational_mass = other.mass * other.gravity_scale
        total += delta.normalize() * (G * gravitational_mass / dist_sq)
    return total


def velocity_verlet_step(bodies: list[Body], dt: float) -> None:
    active = [body for body in bodies if body.alive]
    old_accel = {body.name: acceleration(body, active) for body in active}

    for body in active:
        if body.fixed:
            continue
        body.pos += body.vel * dt + 0.5 * old_accel[body.name] * dt * dt

    new_active = [body for body in bodies if body.alive]
    new_accel = {body.name: acceleration(body, new_active) for body in new_active}

    for body in new_active:
        if body.fixed:
            continue
        body.vel += 0.5 * (old_accel[body.name] + new_accel[body.name]) * dt
        body.record_trail()


def vector_from_angle(angle_degrees: float, speed: float) -> pygame.Vector2:
    radians = math.radians(angle_degrees)
    return pygame.Vector2(math.cos(radians) * speed, math.sin(radians) * speed)


def default_moon_speed() -> float:
    return math.sqrt(G * (PLANET_MASS + MOON_MASS) / MOON_ORBIT_RADIUS)


def make_system(moon_angle: float = 90.0, moon_speed: float | None = None) -> list[Body]:
    moon_speed = default_moon_speed() if moon_speed is None else moon_speed
    relative_moon_velocity = vector_from_angle(moon_angle, moon_speed)
    total_mass = PLANET_MASS + MOON_MASS
    planet_offset = pygame.Vector2(-MOON_ORBIT_RADIUS * MOON_MASS / total_mass, 0)
    moon_offset = pygame.Vector2(MOON_ORBIT_RADIUS * PLANET_MASS / total_mass, 0)

    planet = Body(
        "planet",
        mass=PLANET_MASS,
        radius=34,
        pos=pygame.Vector2(CENTER) + planet_offset,
        vel=-relative_moon_velocity * (MOON_MASS / total_mass),
        color=EARTH_BLUE,
        trail_color=(37, 112, 89),
    )
    moon = Body(
        "moon",
        mass=MOON_MASS,
        radius=11,
        pos=pygame.Vector2(CENTER) + moon_offset,
        vel=relative_moon_velocity * (PLANET_MASS / total_mass),
        color=MOON_GREY,
        trail_color=(97, 111, 134),
    )
    return [planet, moon]


class Simulator:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("The Orbital Mechanic")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)
        self.bodies = make_system()
        self.gravity_scales = {"earth": 1.0, "moon": 1.0, "satellite": 1.0}
        self.gravity_target = "earth"
        self.apply_gravity_scales()
        self.earth_surface = self.create_earth_surface(self.planet.radius)
        self.selected_launch = "satellite"
        self.satellite_angle = 0.0
        self.satellite_speed = 8.45
        self.satellite_vector = pygame.Vector2()
        self.moon_angle = 90.0
        self.moon_speed = default_moon_speed()
        self.moon_vector = pygame.Vector2()
        self.satellite: Body | None = None
        self.paused = False
        self.dragging = False
        self.drag_target = "satellite"
        self.drag_start = pygame.Vector2()
        self.drag_end = pygame.Vector2()
        self.time_index = NORMAL_TIME_INDEX
        self.time_scale = TIME_SCALES[self.time_index]
        self.particles: list[Particle] = []
        self.earth_craters: list[EarthCrater] = []
        self.show_prediction = True
        self.show_hud = True
        self.camera_zoom = 1.0
        self.hud_button_rect = pygame.Rect(WIDTH // 2 - 34, 18, 68, 28)
        self.message = "Satellite selected. Set a vector, then press Space."
        self.update_launch_vectors()

    @property
    def planet(self) -> Body:
        return self.bodies[0]

    def find_body(self, name: str) -> Body | None:
        return next((body for body in self.bodies if body.name == name), None)

    def camera_center(self) -> pygame.Vector2:
        return pygame.Vector2(self.planet.pos)

    def world_to_screen(self, point: pygame.Vector2 | tuple[float, float]) -> pygame.Vector2:
        return (pygame.Vector2(point) - self.camera_center()) * self.camera_zoom + CENTER

    def screen_to_world(self, point: pygame.Vector2 | tuple[float, float]) -> pygame.Vector2:
        return (pygame.Vector2(point) - CENTER) / self.camera_zoom + self.camera_center()

    def scaled_radius(self, radius: float, minimum: int = 1) -> int:
        return max(minimum, round(radius * self.camera_zoom))

    def adjust_camera_zoom(self, wheel_delta: int) -> None:
        if wheel_delta == 0:
            return
        self.camera_zoom *= ZOOM_STEP ** wheel_delta
        self.camera_zoom = max(MIN_CAMERA_ZOOM, min(MAX_CAMERA_ZOOM, self.camera_zoom))
        self.message = f"Camera zoom set to {self.camera_zoom:.2f}x"

    def clone_body(self, body: Body) -> Body:
        return Body(
            name=body.name,
            mass=body.mass,
            radius=body.radius,
            pos=pygame.Vector2(body.pos),
            vel=pygame.Vector2(body.vel),
            color=body.color,
            trail_color=body.trail_color,
            gravity_scale=body.gravity_scale,
            fixed=body.fixed,
            alive=body.alive,
        )

    def gravity_body_name(self, target: str) -> str:
        return "planet" if target == "earth" else target

    def apply_gravity_scales(self) -> None:
        for body in self.bodies:
            if body.name == "planet":
                body.gravity_scale = self.gravity_scales["earth"]
            elif body.name in self.gravity_scales:
                body.gravity_scale = self.gravity_scales[body.name]

    def select_gravity_target(self, target: str) -> None:
        self.gravity_target = target
        self.message = f"Editing {target} gravity: {self.gravity_scales[target]:.2f}x"

    def set_time_index(self, index: int) -> None:
        self.time_index = max(0, min(len(TIME_SCALES) - 1, index))
        self.time_scale = TIME_SCALES[self.time_index]
        self.message = f"Time scale set to {self.time_scale:.2f}x"

    def adjust_time_index(self, direction: int) -> None:
        self.set_time_index(self.time_index + direction)

    def reset_time_scale(self) -> None:
        self.set_time_index(NORMAL_TIME_INDEX)

    def toggle_hud(self) -> None:
        self.show_hud = not self.show_hud
        state = "shown" if self.show_hud else "hidden"
        self.message = f"HUD {state}."

    def time_mode_label(self) -> str:
        if self.time_scale < 1.0:
            return "slow"
        if self.time_scale > 1.0:
            return "fast"
        return "normal"

    def adjust_gravity_scale(self, delta: float) -> None:
        current = self.gravity_scales[self.gravity_target]
        self.gravity_scales[self.gravity_target] = max(0.0, min(3.0, current + delta))
        self.apply_gravity_scales()
        self.message = (
            f"{self.gravity_target.title()} gravity set to "
            f"{self.gravity_scales[self.gravity_target]:.2f}x"
        )

    def reset_gravity_scale(self) -> None:
        self.gravity_scales[self.gravity_target] = 1.0
        self.apply_gravity_scales()
        self.message = f"{self.gravity_target.title()} gravity reset to 1.00x"

    def update_launch_vectors(self) -> None:
        self.satellite_vector = vector_from_angle(
            self.satellite_angle, self.satellite_speed
        )
        self.moon_vector = vector_from_angle(self.moon_angle, self.moon_speed)

    def active_vector(self) -> pygame.Vector2:
        return self.moon_vector if self.selected_launch == "moon" else self.satellite_vector

    def active_color(self) -> tuple[int, int, int]:
        return MOON_GREY if self.selected_launch == "moon" else ORBIT_BLUE

    def select_launch(self, target: str) -> None:
        self.selected_launch = target
        self.message = f"{target.title()} selected. Set a vector, then press Space."

    def satellite_launch_point(self) -> pygame.Vector2:
        outward = pygame.Vector2(0, -1)
        return self.planet.pos + outward * (self.planet.radius + LAUNCH_DISTANCE)

    def moon_launch_point(self) -> pygame.Vector2:
        return self.planet.pos + pygame.Vector2(MOON_ORBIT_RADIUS, 0)

    def active_launch_point(self) -> pygame.Vector2:
        if self.selected_launch == "moon":
            moon = self.find_body("moon")
            if moon and moon.alive:
                return pygame.Vector2(moon.pos)
            return self.moon_launch_point()
        return self.satellite_launch_point()

    def predicted_body(self) -> Body:
        if self.selected_launch == "moon":
            return Body(
                "moon",
                mass=MOON_MASS,
                radius=11,
                pos=self.active_launch_point(),
                vel=self.planet.vel + self.moon_vector,
                color=MOON_GREY,
                trail_color=(97, 111, 134),
                gravity_scale=self.gravity_scales["moon"],
            )

        return Body(
            "satellite",
            mass=SATELLITE_MASS,
            radius=5,
            pos=self.satellite_launch_point(),
            vel=self.planet.vel + self.satellite_vector,
            color=SAT_GOLD,
            trail_color=(194, 139, 38),
            gravity_scale=self.gravity_scales["satellite"],
        )

    def prediction_path(self) -> list[tuple[int, int]]:
        if not self.show_prediction:
            return []

        target = self.predicted_body()
        prediction_bodies = [
            self.clone_body(body)
            for body in self.bodies
            if body.alive and body.name != target.name
        ]
        prediction_bodies.append(target)
        points: list[tuple[int, int]] = [(round(target.pos.x), round(target.pos.y))]

        for step in range(PREDICTION_STEPS):
            velocity_verlet_step(prediction_bodies, TIME_STEP)
            if step % PREDICTION_SAMPLE_EVERY == 0:
                points.append((round(target.pos.x), round(target.pos.y)))

            for body in prediction_bodies:
                if body is target or not body.alive:
                    continue
                if target.pos.distance_to(body.pos) <= target.radius + body.radius:
                    return points

            if (
                target.pos.x < -PREDICTION_MARGIN
                or target.pos.x > WIDTH + PREDICTION_MARGIN
                or target.pos.y < -PREDICTION_MARGIN
                or target.pos.y > HEIGHT + PREDICTION_MARGIN
            ):
                return points

        return points

    def is_clicking_moon(self, mouse_world_pos: pygame.Vector2) -> bool:
        moon = self.find_body("moon")
        if not moon or not moon.alive:
            return False
        return moon.pos.distance_to(mouse_world_pos) <= moon.radius + 14 / self.camera_zoom

    def launch_satellite(self) -> None:
        if self.satellite and self.satellite in self.bodies:
            self.bodies.remove(self.satellite)

        self.satellite = Body(
            "satellite",
            mass=SATELLITE_MASS,
            radius=5,
            pos=self.satellite_launch_point(),
            vel=self.planet.vel + self.satellite_vector,
            color=SAT_GOLD,
            trail_color=(194, 139, 38),
            gravity_scale=self.gravity_scales["satellite"],
        )
        self.bodies.append(self.satellite)
        self.message = (
            f"Satellite launched at vx={self.satellite_vector.x:.2f}, "
            f"vy={self.satellite_vector.y:.2f}"
        )

    def launch_moon(self) -> None:
        self.bodies = [body for body in self.bodies if body.name != "moon"]
        moon = Body(
            "moon",
            mass=MOON_MASS,
            radius=11,
            pos=self.moon_launch_point(),
            vel=self.planet.vel + self.moon_vector,
            color=MOON_GREY,
            trail_color=(97, 111, 134),
            gravity_scale=self.gravity_scales["moon"],
        )
        self.bodies.append(moon)
        self.message = (
            f"Moon launched at vx={self.moon_vector.x:.2f}, "
            f"vy={self.moon_vector.y:.2f}"
        )

    def launch_selected(self) -> None:
        if self.selected_launch == "moon":
            self.launch_moon()
        else:
            self.launch_satellite()

    def apply_moon_velocity(self) -> None:
        moon = self.find_body("moon")
        if not moon or not moon.alive:
            self.launch_moon()
            return

        moon.vel = self.planet.vel + self.moon_vector
        moon.trail.clear()
        self.message = (
            f"Moon velocity changed to vx={self.moon_vector.x:.2f}, "
            f"vy={self.moon_vector.y:.2f}"
        )

    def reset(self) -> None:
        self.bodies = make_system(self.moon_angle, self.moon_speed)
        self.apply_gravity_scales()
        self.satellite = None
        self.particles.clear()
        self.earth_craters.clear()
        self.message = "System reset with the current moon vector."

    def handle_key(self, key: int, mods: int) -> None:
        fine = 0.25 if mods & pygame.KMOD_SHIFT else 1.0
        speed_step = 0.2 * fine
        angle_step = 2.0 * fine
        gravity_step = 0.20 if mods & pygame.KMOD_SHIFT else 0.05
        time_step = 2 if mods & pygame.KMOD_SHIFT else 1

        if key == pygame.K_SPACE:
            self.launch_selected()
        elif key == pygame.K_TAB:
            self.select_launch("moon" if self.selected_launch == "satellite" else "satellite")
        elif key == pygame.K_s:
            self.select_launch("satellite")
        elif key == pygame.K_m:
            self.select_launch("moon")
        elif key == pygame.K_1:
            self.select_gravity_target("earth")
        elif key == pygame.K_2:
            self.select_gravity_target("moon")
        elif key == pygame.K_3:
            self.select_gravity_target("satellite")
        elif key == pygame.K_LEFTBRACKET:
            self.adjust_gravity_scale(-gravity_step)
        elif key == pygame.K_RIGHTBRACKET:
            self.adjust_gravity_scale(gravity_step)
        elif key == pygame.K_0:
            self.reset_gravity_scale()
        elif key == pygame.K_g:
            self.show_prediction = not self.show_prediction
            state = "shown" if self.show_prediction else "hidden"
            self.message = f"Orbit prediction {state}."
        elif key == pygame.K_h:
            self.toggle_hud()
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_c:
            for body in self.bodies:
                body.trail.clear()
            self.message = "Trails cleared."
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_PERIOD):
            self.adjust_time_index(time_step)
        elif key in (pygame.K_MINUS, pygame.K_COMMA):
            self.adjust_time_index(-time_step)
        elif key == pygame.K_t:
            self.reset_time_scale()
        elif key == pygame.K_n:
            if self.paused:
                self.step_paused_frame()
            else:
                self.message = "Press P to pause, then N to step one frame."
        elif key == pygame.K_LEFT:
            if self.selected_launch == "moon":
                self.moon_angle -= angle_step
            else:
                self.satellite_angle -= angle_step
        elif key == pygame.K_RIGHT:
            if self.selected_launch == "moon":
                self.moon_angle += angle_step
            else:
                self.satellite_angle += angle_step
        elif key == pygame.K_UP:
            if self.selected_launch == "moon":
                self.moon_speed = min(18.0, self.moon_speed + speed_step)
            else:
                self.satellite_speed = min(18.0, self.satellite_speed + speed_step)
        elif key == pygame.K_DOWN:
            if self.selected_launch == "moon":
                self.moon_speed = max(0.0, self.moon_speed - speed_step)
            else:
                self.satellite_speed = max(0.0, self.satellite_speed - speed_step)

        self.update_launch_vectors()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                self.handle_key(event.key, pygame.key.get_mods())
            if event.type == pygame.MOUSEWHEEL:
                self.adjust_camera_zoom(event.y)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.hud_button_rect.collidepoint(event.pos):
                    self.toggle_hud()
                    continue
                mouse_world = self.screen_to_world(event.pos)
                if self.is_clicking_moon(mouse_world):
                    self.select_launch("moon")
                    self.drag_target = "moon"
                else:
                    self.drag_target = self.selected_launch
                self.dragging = True
                self.drag_start = self.active_launch_point()
                self.drag_end = mouse_world
            if event.type == pygame.MOUSEMOTION and self.dragging:
                self.drag_end = self.screen_to_world(event.pos)
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
                self.dragging = False
                vector = (self.screen_to_world(event.pos) - self.drag_start) * DRAG_TO_VELOCITY
                if vector.length() > 0.1:
                    if self.drag_target == "moon":
                        self.moon_vector = vector
                        self.moon_speed = vector.length()
                        self.moon_angle = math.degrees(math.atan2(vector.y, vector.x))
                        self.apply_moon_velocity()
                    else:
                        self.satellite_vector = vector
                        self.satellite_speed = vector.length()
                        self.satellite_angle = math.degrees(math.atan2(vector.y, vector.x))
                        self.message = (
                            f"Satellite vector set by drag: "
                            f"vx={vector.x:.2f}, vy={vector.y:.2f}"
                        )
        return True

    def update(self) -> None:
        if self.paused or (self.dragging and self.drag_target == "moon"):
            return

        self.advance_physics_frame()

    def advance_physics_frame(self) -> None:
        for _ in range(4):
            velocity_verlet_step(self.bodies, TIME_STEP * self.time_scale)
            self.check_satellite_status()
            self.check_moon_status()
            self.emit_moon_flame_streak()
        self.update_particles()

    def step_paused_frame(self) -> None:
        if self.dragging and self.drag_target == "moon":
            return
        self.advance_physics_frame()
        self.message = f"Stepped one frame at {self.time_scale:.2f}x"

    def update_particles(self) -> None:
        for particle in self.particles:
            particle.update(self.time_scale)
        self.particles = [particle for particle in self.particles if particle.alive]
        if len(self.particles) > MAX_PARTICLES:
            self.particles = self.particles[-MAX_PARTICLES:]

    def emit_moon_flame_streak(self) -> None:
        moon = self.find_body("moon")
        if not moon or not moon.alive:
            return

        distance = moon.pos.distance_to(self.planet.pos)
        collision_distance = self.planet.radius + moon.radius
        if distance > MOON_FLAME_DISTANCE or distance <= collision_distance:
            return

        relative_velocity = moon.vel - self.planet.vel
        if relative_velocity.length_squared() < 1.0:
            return

        heat = 1.0 - (distance - collision_distance) / (MOON_FLAME_DISTANCE - collision_distance)
        heat = max(0.0, min(1.0, heat))
        flight_dir = relative_velocity.normalize()
        tail_dir = -flight_dir
        side_dir = flight_dir.rotate(90)

        fire_count = max(1, int(2 + heat * 7))
        smoke_count = max(1, int(1 + heat * 4))

        for _ in range(fire_count):
            self.particles.append(
                Particle(
                    pos=moon.pos - flight_dir * random.uniform(2, moon.radius + 4)
                    + side_dir * random.uniform(-moon.radius * 0.35, moon.radius * 0.35),
                    vel=tail_dir * random.uniform(1.7, 4.6)
                    + side_dir * random.uniform(-1.2, 1.2),
                    radius=random.uniform(1.5, 3.6 + heat * 2.4),
                    color=random.choice(FIRE_COLORS),
                    lifetime=random.uniform(16, 32),
                    growth=random.uniform(-0.03, 0.025),
                    drag=random.uniform(0.90, 0.965),
                )
            )

        for _ in range(smoke_count):
            self.particles.append(
                Particle(
                    pos=moon.pos + tail_dir * random.uniform(moon.radius * 0.4, moon.radius * 1.6)
                    + side_dir * random.uniform(-moon.radius * 0.55, moon.radius * 0.55),
                    vel=tail_dir * random.uniform(0.7, 2.4)
                    + side_dir * random.uniform(-0.9, 0.9),
                    radius=random.uniform(3.2, 6.0 + heat * 4.0),
                    color=random.choice(SMOKE_COLORS),
                    lifetime=random.uniform(40, 78),
                    growth=random.uniform(0.035, 0.09),
                    drag=random.uniform(0.965, 0.988),
                )
            )

    def spawn_impact(self, incoming: Body, impact_scale: float = 1.0) -> None:
        normal = incoming.pos - self.planet.pos
        if normal.length_squared() == 0:
            normal = pygame.Vector2(0, -1)
        else:
            normal = normal.normalize()

        tangent = normal.rotate(90)
        impact_pos = self.planet.pos + normal * (self.planet.radius + 2)
        incoming_dir = incoming.vel - self.planet.vel
        if incoming_dir.length_squared() > 0:
            incoming_dir = incoming_dir.normalize()
        else:
            incoming_dir = -normal

        fire_count = int(42 * impact_scale)
        smoke_count = int(30 * impact_scale)

        for _ in range(fire_count):
            spread = tangent * random.uniform(-2.8, 2.8)
            outward = normal * random.uniform(2.2, 6.3)
            rebound = -incoming_dir * random.uniform(0.2, 2.4)
            self.particles.append(
                Particle(
                    pos=impact_pos + normal * random.uniform(0, 5),
                    vel=outward + spread + rebound,
                    radius=random.uniform(2.0, 4.8),
                    color=random.choice(FIRE_COLORS),
                    lifetime=random.uniform(18, 34),
                    growth=random.uniform(-0.04, 0.02),
                    drag=random.uniform(0.90, 0.96),
                )
            )

        for _ in range(smoke_count):
            spread = tangent * random.uniform(-1.9, 1.9)
            outward = normal * random.uniform(0.8, 2.8)
            self.particles.append(
                Particle(
                    pos=impact_pos + normal * random.uniform(2, 9),
                    vel=outward + spread + pygame.Vector2(0, random.uniform(-0.35, 0.18)),
                    radius=random.uniform(4.0, 8.0),
                    color=random.choice(SMOKE_COLORS),
                    lifetime=random.uniform(54, 96),
                    growth=random.uniform(0.045, 0.11),
                    drag=random.uniform(0.965, 0.987),
                )
            )

    def create_earth_crater(self, incoming: Body) -> None:
        normal = incoming.pos - self.planet.pos
        if normal.length_squared() == 0:
            normal = pygame.Vector2(0, -1)
        else:
            normal = normal.normalize()

        crater_radius = max(15.0, min(self.planet.radius * 0.58, incoming.radius * 1.7))
        self.earth_craters.append(EarthCrater(normal=normal, radius=crater_radius))
        self.earth_craters = self.earth_craters[-4:]

    def check_satellite_status(self) -> None:
        if not self.satellite or not self.satellite.alive:
            return

        for body in self.bodies:
            if body is self.satellite or not body.alive:
                continue
            if self.satellite.pos.distance_to(body.pos) <= body.radius + self.satellite.radius:
                self.satellite.alive = False
                self.message = f"Satellite impacted the {body.name}."
                if body is self.planet:
                    self.spawn_impact(self.satellite, impact_scale=1.0)

        margin = 1200
        if (
            self.satellite.pos.x < -margin
            or self.satellite.pos.x > WIDTH + margin
            or self.satellite.pos.y < -margin
            or self.satellite.pos.y > HEIGHT + margin
        ):
            self.satellite.alive = False
            self.message = "Satellite escaped the local system."

    def check_moon_status(self) -> None:
        moon = self.find_body("moon")
        if not moon or not moon.alive:
            return

        if moon.pos.distance_to(self.planet.pos) <= self.planet.radius + moon.radius:
            moon.alive = False
            self.message = "Moon impacted the planet."
            self.create_earth_crater(moon)
            self.spawn_impact(moon, impact_scale=1.8)

        margin = 1200
        if (
            moon.pos.x < -margin
            or moon.pos.x > WIDTH + margin
            or moon.pos.y < -margin
            or moon.pos.y > HEIGHT + margin
        ):
            moon.alive = False
            self.message = "Moon escaped the local system."

    def draw_grid(self) -> None:
        spacing = 50
        top_left = self.screen_to_world((0, 0))
        bottom_right = self.screen_to_world((WIDTH, HEIGHT))
        start_x = math.floor(top_left.x / spacing) * spacing
        end_x = math.ceil(bottom_right.x / spacing) * spacing
        start_y = math.floor(top_left.y / spacing) * spacing
        end_y = math.ceil(bottom_right.y / spacing) * spacing

        x = start_x
        while x <= end_x:
            screen_x = round(self.world_to_screen((x, 0)).x)
            pygame.draw.line(self.screen, GRID, (screen_x, 0), (screen_x, HEIGHT), 1)
            x += spacing

        y = start_y
        while y <= end_y:
            screen_y = round(self.world_to_screen((0, y)).y)
            pygame.draw.line(self.screen, GRID, (0, screen_y), (WIDTH, screen_y), 1)
            y += spacing

    def draw_trails(self) -> None:
        for body in self.bodies:
            if len(body.trail) < 3:
                continue
            points = [
                (round(screen.x), round(screen.y))
                for screen in (self.world_to_screen(point) for point in body.trail)
            ]
            pygame.draw.lines(self.screen, body.trail_color, False, points, 2)

    def draw_prediction(self) -> None:
        points = self.prediction_path()
        if len(points) < 2:
            return

        ghost_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        screen_points = [
            (round(screen.x), round(screen.y))
            for screen in (self.world_to_screen(point) for point in points)
        ]
        for index in range(1, len(points)):
            if index % 2 == 0:
                continue
            alpha = max(34, 132 - index)
            pygame.draw.line(
                ghost_surface,
                (*GHOST, alpha),
                screen_points[index - 1],
                screen_points[index],
                2,
            )

        for index, point in enumerate(screen_points[::5]):
            alpha = max(28, 115 - index * 5)
            pygame.draw.circle(ghost_surface, (*GHOST, alpha), point, 2)

        self.screen.blit(ghost_surface, (0, 0))

    def draw_vector_arrow(
        self,
        start: pygame.Vector2,
        vector: pygame.Vector2,
        color: tuple[int, int, int],
        scale: float = 14.0,
    ) -> None:
        start_screen = self.world_to_screen(start)
        end_screen = self.world_to_screen(start + vector * scale)
        pygame.draw.line(self.screen, color, start_screen, end_screen, 3)
        if vector.length() == 0:
            return

        direction = end_screen - start_screen
        if direction.length_squared() == 0:
            return
        direction = direction.normalize()
        left = direction.rotate(145) * 14
        right = direction.rotate(-145) * 14
        pygame.draw.polygon(self.screen, color, [end_screen, end_screen + left, end_screen + right])

    def create_earth_surface(self, radius: int) -> pygame.Surface:
        size = radius * 2 + 8
        center = pygame.Vector2(size / 2, size / 2)
        surface = pygame.Surface((size, size), pygame.SRCALPHA)

        for y in range(size):
            for x in range(size):
                offset = pygame.Vector2(x + 0.5, y + 0.5) - center
                distance = offset.length()
                if distance > radius:
                    continue

                nx = offset.x / radius
                ny = offset.y / radius
                light = max(0.0, min(1.0, 0.82 - 0.23 * nx - 0.28 * ny))
                edge = max(0.0, min(1.0, 1.0 - distance / radius))
                ocean = (
                    int(EARTH_BLUE[0] * light + 7 * edge),
                    int(EARTH_BLUE[1] * light + 22 * edge),
                    int(EARTH_BLUE[2] * light + 35 * edge),
                    255,
                )
                surface.set_at((x, y), ocean)

        land_shapes = [
            [(-0.62, -0.45), (-0.30, -0.62), (-0.08, -0.42), (-0.17, -0.12), (-0.47, -0.03), (-0.67, -0.20)],
            [(-0.30, 0.00), (-0.08, 0.13), (-0.02, 0.42), (-0.18, 0.72), (-0.39, 0.50), (-0.45, 0.19)],
            [(0.06, -0.45), (0.42, -0.57), (0.74, -0.35), (0.68, -0.05), (0.28, 0.01), (0.02, -0.18)],
            [(0.23, 0.03), (0.48, 0.11), (0.58, 0.35), (0.43, 0.63), (0.17, 0.51), (0.09, 0.22)],
            [(0.52, 0.55), (0.73, 0.61), (0.78, 0.78), (0.58, 0.83), (0.43, 0.70)],
        ]
        for shape in land_shapes:
            points = [
                (center.x + x * radius, center.y + y * radius)
                for x, y in shape
            ]
            pygame.draw.polygon(surface, EARTH_LAND, points)

        pygame.draw.ellipse(
            surface,
            EARTH_ICE,
            pygame.Rect(center.x - radius * 0.38, center.y - radius * 0.98, radius * 0.76, radius * 0.18),
        )
        pygame.draw.ellipse(
            surface,
            EARTH_ICE,
            pygame.Rect(center.x - radius * 0.48, center.y + radius * 0.82, radius * 0.96, radius * 0.18),
        )

        cloud_color = (246, 250, 255, 170)
        cloud_streaks = [
            (-0.65, -0.18, 0.45, 0.10),
            (-0.16, -0.33, 0.52, 0.11),
            (0.18, 0.18, 0.46, 0.10),
            (-0.50, 0.42, 0.36, 0.09),
        ]
        for x, y, width, height in cloud_streaks:
            rect = pygame.Rect(
                center.x + x * radius,
                center.y + y * radius,
                width * radius,
                height * radius,
            )
            pygame.draw.ellipse(surface, cloud_color, rect)

        pygame.draw.circle(surface, (255, 255, 255, 88), center, radius, 1)
        pygame.draw.circle(surface, (6, 18, 44, 80), center, radius, 3)
        return surface

    def draw_earth(self, pos: pygame.Vector2) -> None:
        earth = self.earth_surface.copy()
        for crater in self.earth_craters:
            self.draw_earth_crater(earth, crater)

        if self.camera_zoom != 1.0:
            scaled_size = (
                max(4, round(earth.get_width() * self.camera_zoom)),
                max(4, round(earth.get_height() * self.camera_zoom)),
            )
            earth = pygame.transform.smoothscale(earth, scaled_size)

        screen_pos = self.world_to_screen(pos)
        rect = earth.get_rect(center=(round(screen_pos.x), round(screen_pos.y)))
        self.screen.blit(earth, rect)

    def crater_polygon(
        self,
        center: pygame.Vector2,
        normal: pygame.Vector2,
        radius: float,
        flatten: float,
        samples: int = 28,
    ) -> list[tuple[float, float]]:
        tangent = normal.rotate(90)
        points = []
        for index in range(samples):
            angle = math.tau * index / samples
            point = (
                center
                + tangent * math.cos(angle) * radius
                + normal * math.sin(angle) * radius * flatten
            )
            points.append((point.x, point.y))
        return points

    def draw_earth_crater(self, surface: pygame.Surface, crater: EarthCrater) -> None:
        center = pygame.Vector2(surface.get_width() / 2, surface.get_height() / 2)
        normal = pygame.Vector2(crater.normal)
        if normal.length_squared() == 0:
            normal = pygame.Vector2(0, -1)
        else:
            normal = normal.normalize()

        crater_center = center + normal * (self.planet.radius * 0.66)
        bite_center = center + normal * (self.planet.radius * 1.02)
        bite_radius = max(6, int(crater.radius * 0.72))
        pygame.draw.circle(surface, (0, 0, 0, 0), bite_center, bite_radius)

        glow_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for scale, alpha in [(1.45, 58), (1.05, 95), (0.66, 150)]:
            pygame.draw.polygon(
                glow_surface,
                (*CRATER_GLOW, int(alpha * crater.glow)),
                self.crater_polygon(crater_center, normal, crater.radius * scale, 0.42),
            )
        surface.blit(glow_surface, (0, 0))

        pygame.draw.polygon(
            surface,
            CRATER_RIM,
            self.crater_polygon(crater_center, normal, crater.radius * 1.08, 0.45),
        )
        pygame.draw.polygon(
            surface,
            CRATER_DARK,
            self.crater_polygon(crater_center + normal * 1.5, normal, crater.radius * 0.78, 0.38),
        )
        pygame.draw.polygon(
            surface,
            CRATER_GLOW,
            self.crater_polygon(crater_center - normal * 1.0, normal, crater.radius * 0.48, 0.30),
        )
        pygame.draw.circle(
            surface,
            CRATER_CORE,
            crater_center - normal * (crater.radius * 0.08),
            max(2, int(crater.radius * 0.18)),
        )

    def draw_bodies(self) -> None:
        for body in self.bodies:
            if not body.alive:
                continue
            if body.name == "planet":
                self.draw_earth(body.pos)
            else:
                screen_pos = self.world_to_screen(body.pos)
                radius = self.scaled_radius(body.radius, minimum=2)
                pygame.draw.circle(self.screen, body.color, screen_pos, radius)
                pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, radius, 1)

        if not self.show_hud:
            return

        satellite_launch = self.satellite_launch_point()
        moon_launch = self.active_launch_point() if self.selected_launch == "moon" else self.moon_launch_point()
        satellite_screen = self.world_to_screen(satellite_launch)
        moon_screen = self.world_to_screen(moon_launch)
        pygame.draw.circle(self.screen, (255, 255, 255), satellite_screen, 4)
        if self.selected_launch != "moon":
            pygame.draw.circle(self.screen, MOON_GREY, moon_screen, 4)
        self.draw_vector_arrow(satellite_launch, self.satellite_vector, ORBIT_BLUE)
        self.draw_vector_arrow(moon_launch, self.moon_vector, MOON_GREY)

        active_point = self.active_launch_point()
        active_screen = self.world_to_screen(active_point)
        pygame.draw.circle(self.screen, self.active_color(), active_screen, 9, 2)

        if self.dragging:
            drag_vector = (self.drag_end - self.drag_start) * DRAG_TO_VELOCITY
            self.draw_vector_arrow(self.drag_start, drag_vector, SAT_GOLD)

    def draw_particles(self) -> None:
        for particle in sorted(self.particles, key=lambda item: item.radius, reverse=True):
            fade = max(0.0, 1.0 - particle.age / particle.lifetime)
            alpha = int(230 * fade)
            radius = max(1, particle.radius * self.camera_zoom)
            size = max(2, int(radius * 2 + 4))
            particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            center = (size // 2, size // 2)
            pygame.draw.circle(
                particle_surface,
                (*particle.color, alpha),
                center,
                max(1, int(radius)),
            )
            screen_pos = self.world_to_screen(particle.pos)
            self.screen.blit(
                particle_surface,
                (screen_pos.x - center[0], screen_pos.y - center[1]),
            )

    def draw_hud_button(self) -> None:
        button_surface = pygame.Surface(
            (self.hud_button_rect.width, self.hud_button_rect.height),
            pygame.SRCALPHA,
        )
        alpha = 190 if self.show_hud else 70
        pygame.draw.rect(
            button_surface,
            (*PANEL, alpha),
            button_surface.get_rect(),
            border_radius=7,
        )
        pygame.draw.rect(
            button_surface,
            (96, 112, 138, alpha),
            button_surface.get_rect(),
            1,
            border_radius=7,
        )
        label = "Hide" if self.show_hud else "HUD"
        text = self.small_font.render(label, True, TEXT if self.show_hud else MUTED)
        button_surface.blit(
            text,
            (
                button_surface.get_width() // 2 - text.get_width() // 2,
                button_surface.get_height() // 2 - text.get_height() // 2,
            ),
        )
        self.screen.blit(button_surface, self.hud_button_rect)

    def text(self, value: str, x: int, y: int, color: tuple[int, int, int] = TEXT) -> None:
        self.screen.blit(self.font.render(value, True, color), (x, y))

    def small_text(
        self, value: str, x: int, y: int, color: tuple[int, int, int] = MUTED
    ) -> None:
        self.screen.blit(self.small_font.render(value, True, color), (x, y))

    def draw_panel(self) -> None:
        panel_rect = pygame.Rect(18, 18, 430, 342)
        pygame.draw.rect(self.screen, PANEL, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 66, 88), panel_rect, 1, border_radius=8)

        self.screen.blit(self.title_font.render("The Orbital Mechanic", True, TEXT), (34, 32))
        self.small_text("N-body gravity with launch vectors", 35, 63)

        self.text(f"selected: {self.selected_launch}", 35, 90, self.active_color())
        prediction_state = "on" if self.show_prediction else "off"
        self.text(f"t:{self.time_scale:4.2f}x z:{self.camera_zoom:.2f}", 230, 90)
        self.small_text(
            f"gravity target: {self.gravity_target}",
            35,
            120,
            WARNING,
        )
        self.small_text(
            f"earth {self.gravity_scales['earth']:.2f}x   "
            f"moon {self.gravity_scales['moon']:.2f}x   "
            f"sat {self.gravity_scales['satellite']:.2f}x",
            35,
            142,
            TEXT,
        )
        self.small_text("Satellite vector", 35, 170, ORBIT_BLUE)
        self.text(
            f"vx {self.satellite_vector.x:6.2f}  vy {self.satellite_vector.y:6.2f}",
            35,
            193,
            ORBIT_BLUE,
        )
        self.text(
            f"speed {self.satellite_speed:5.2f}  angle {self.satellite_angle:6.1f}",
            35,
            218,
        )
        self.small_text("Moon vector", 35, 246, MOON_GREY)
        self.text(
            f"vx {self.moon_vector.x:6.2f}  vy {self.moon_vector.y:6.2f}",
            35,
            269,
            MOON_GREY,
        )
        self.text(
            f"speed {self.moon_speed:5.2f}  angle {self.moon_angle:6.1f}",
            35,
            294,
        )
        moon = self.find_body("moon")
        lost_satellite = self.satellite and not self.satellite.alive
        lost_moon = moon and not moon.alive
        status_color = WARNING if lost_satellite or lost_moon else MUTED
        self.small_text(self.message, 35, 326, status_color)

        help_x = WIDTH - 388
        help_rect = pygame.Rect(help_x, 18, 370, 328)
        pygame.draw.rect(self.screen, PANEL, help_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 66, 88), help_rect, 1, border_radius=8)
        self.text("Controls", help_x + 18, 34)
        controls = [
            "H / Hide button: toggle HUD",
            "Mouse wheel: zoom view",
            "Tab: switch satellite/moon",
            "Click moon: edit its live velocity",
            "S / M: select satellite or moon",
            "1/2/3: edit Earth/moon/sat gravity",
            "[ / ]: lower/raise gravity",
            "0: reset selected gravity",
            "G: toggle ghost prediction",
            ", / . or +/-: scrub time speed",
            "T: normal time   N: step paused",
            "Left/Right: rotate launch vector",
            "Up/Down: adjust launch speed",
            "Shift + arrows: fine adjustment",
            "Drag from active point: set vx/vy",
            "Space: relaunch selected body",
            "P: pause   R: reset   C: clear trails",
        ]
        for index, line in enumerate(controls):
            self.small_text(line, help_x + 18, 68 + index * 15)

    def draw_energy_readout(self) -> None:
        y = HEIGHT - 54
        moon = self.find_body("moon")
        if moon and moon.alive:
            self.draw_orbit_readout("moon", moon, y, MOON_GREY)
            y += 22
        if self.satellite and self.satellite.alive:
            self.draw_orbit_readout("sat", self.satellite, y, SAT_GOLD)

    def draw_orbit_readout(
        self, label: str, body: Body, y: int, color: tuple[int, int, int]
    ) -> None:
        distance = body.pos.distance_to(self.planet.pos)
        relative_speed = (body.vel - self.planet.vel).length()
        earth_gravity = self.planet.mass * self.planet.gravity_scale
        specific_energy = 0.5 * relative_speed * relative_speed - G * earth_gravity / max(distance, 1)
        orbit_type = "bound ellipse" if specific_energy < 0 else "escape path"
        self.small_text(
            f"{label} r={distance:6.1f}  v={relative_speed:5.2f}  energy={specific_energy:6.2f}  {orbit_type}",
            24,
            y,
            color,
        )

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self.draw_grid()
        if self.show_hud:
            self.draw_prediction()
        self.draw_trails()
        self.draw_bodies()
        self.draw_particles()
        if self.show_hud:
            self.draw_panel()
            self.draw_energy_readout()
        self.draw_hud_button()
        if self.paused and self.show_hud:
            paused = self.title_font.render("PAUSED", True, WARNING)
            self.screen.blit(paused, (WIDTH // 2 - paused.get_width() // 2, 30))
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    Simulator().run()
