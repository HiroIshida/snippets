import pymunk
import pymunk.pygame_util
import pygame
import random

def create_ball(space, position, radius=1.5, mass=1):
    body = pymunk.Body()
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.color = pygame.Color("red")
    shape.mass = mass
    shape.elasticity = 0.0
    shape.friction = 0.0
    shape.collision_type = 1
    space.add(body, shape)
    return shape

def create_saucer(space, position, size=(200, 50)):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = position
    left = pymunk.Segment(body, (-size[0]/2, -size[1]/2), (-size[0]/2, size[1]/2), 5)
    right = pymunk.Segment(body, (size[0]/2, -size[1]/2), (size[0]/2, size[1]/2), 5)
    bottom = pymunk.Segment(body, (-size[0]/2, size[1]/2), (size[0]/2, size[1]/2), 8)
    left.elasticity = right.elasticity = bottom.elasticity = 0.0
    left.friction = right.friction = bottom.friction = 0.2
    space.add(body, left, right, bottom)

def create_cup(space, position, size=(80, 100)):
    body = pymunk.Body(1000, float("inf"))
    body.position = position
    left = pymunk.Segment(body, (-size[0]/2, -size[1]/2), (-size[0]/2, size[1]/2), 5)
    right = pymunk.Segment(body, (size[0]/2, -size[1]/2), (size[0]/2, size[1]/2), 5)
    bottom = pymunk.Segment(body, (-size[0]/2, size[1]/2), (size[0]/2, size[1]/2), 5)
    left.elasticity = right.elasticity = bottom.elasticity = 0.0
    left.friction = right.friction = bottom.friction = 0.2
    space.add(body, left, right, bottom)
    return body

def main():
    pygame.init()
    font = pygame.font.SysFont("Arial", 24)
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Pymunk Ball and Cup Simulation")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    space.gravity = (0, 900)

    for _ in range(600):
        position = (random.randint(289, 320), random.randint(150, 300))
        create_ball(space, position)



    pos = (300, 400)
    body = create_cup(space, pos)
    static_body = space.static_body
    pivot_joint = pymunk.PivotJoint(static_body, body, pos)
    space.add(pivot_joint)

    for _ in range(100):
      space.step(1/60.0)

    create_saucer(space, (200, 550))

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                body.angle -= 0.01
            if event.key == pygame.K_RIGHT:
                body.angle += 0.01

        space.step(1/60.0)
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)

        saucer_y_min = 550 - 25
        saucer_y_max = 550 + 25
        count = -3  # -3 because there are 3 static segments in the saucer
        for shape in space.shapes:
            if shape.body.position.y > saucer_y_min and shape.body.position.y < saucer_y_max:
                count += 1

        text_surface = font.render(f"Count: {count}", True, pygame.Color("black"))
        screen.blit(text_surface, (300 - text_surface.get_width() // 2, 300 - text_surface.get_height() // 2))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
