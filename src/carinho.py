import math
import pygame
import matplotlib.pyplot as plt

def sinc_safe(z: float) -> float:
    if abs(z) < 1e-9:
        return 1.0 - z * z / 6.0
    return math.sin(z) / z


WORLD_X_MIN, WORLD_X_MAX = -1.0, 5.0
WORLD_Y_MIN, WORLD_Y_MAX = -2.0, 3.0
FINISH_X = 4.0
FINISH_Y_RANGE = (WORLD_Y_MIN, WORLD_Y_MAX)
ROBOT_RADIUS = 0.06

PARKED_CARS = [
    (1.2, -1.0, 0.3, 0.2),
    (2.5, -1.5, 0.3, 0.2),
    (3.8,  0.5, 0.3, 0.2),
    (0.5,  1.5, 0.3, 0.2),
    (2.0,  2.0, 0.3, 0.2),
]


def circle_rect_collision(cx, cy, radius, rect):
    rx, ry, rw, rh = rect
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    dist_sq = (cx - closest_x)*2 + (cy - closest_y)*2
    return dist_sq <= radius**2


def interactive_simulation(r, b, dt, x0, y0, theta0):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robô – Setas: andar/curvar   ESPAÇO: parar   ↓: girar 180°")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)


    x, y, theta = float(x0), float(y0), float(theta0)
    trail = [(x, y)]
    prev_x = x0

    base_speed = 0.4            
    turn_factor = 0.5        
    command = 'stop'           


    cam_x, cam_y = x, y
    scale = 150


    BG_COLOR = (30, 30, 30)
    TRAIL_COLOR = (0, 200, 0)
    ROBOT_COLOR = (255, 100, 100)
    WALL_COLOR = (200, 200, 200)
    FINISH_COLOR = (255, 255, 0)
    CAR_COLOR = (100, 100, 255)

    running = True
    finish_crossed = False
    collision = False

    while running:
        clock.tick(int(1 / dt))
        dt_actual = dt

      
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
          
                elif event.key == pygame.K_UP:
                    command = 'straight'
                elif event.key == pygame.K_LEFT:
                    command = 'left'
                elif event.key == pygame.K_RIGHT:
                    command = 'right'
      
                elif event.key == pygame.K_SPACE:
                    command = 'stop'
            
                elif event.key == pygame.K_DOWN:
                    command = 'stop'
                    theta += math.pi
                    theta = (theta + math.pi) % (2 * math.pi) - math.pi  
                 
                    trail.append((x, y))  

        if command == 'stop':
            current_speed = 0.0
            turn_dir = 0.0
        else:
            current_speed = base_speed
            if command == 'straight':
                turn_dir = 0.0
            elif command == 'left':
                turn_dir = 1.0
            elif command == 'right':
                turn_dir = -1.0
            else:
                turn_dir = 0.0 

      
        w_r = (current_speed / r) * (1.0 + turn_dir * turn_factor)
        w_l = (current_speed / r) * (1.0 - turn_dir * turn_factor)


        v = r * (w_r + w_l) / 2.0
        omega = r * (w_r - w_l) / (2.0 * b)
        alpha = 0.5 * dt_actual * omega
        s = sinc_safe(alpha)
        delta_x = dt_actual * v * s * math.cos(theta + alpha)
        delta_y = dt_actual * v * s * math.sin(theta + alpha)
        delta_theta = dt_actual * omega

 
        new_x = x + delta_x
        new_y = y + delta_y

        if new_x - ROBOT_RADIUS < WORLD_X_MIN:
            new_x = WORLD_X_MIN + ROBOT_RADIUS
            delta_x = new_x - x
        if new_x + ROBOT_RADIUS > WORLD_X_MAX:
            new_x = WORLD_X_MAX - ROBOT_RADIUS
            delta_x = new_x - x
        if new_y - ROBOT_RADIUS < WORLD_Y_MIN:
            new_y = WORLD_Y_MIN + ROBOT_RADIUS
            delta_y = new_y - y
        if new_y + ROBOT_RADIUS > WORLD_Y_MAX:
            new_y = WORLD_Y_MAX - ROBOT_RADIUS
            delta_y = new_y - y

        x += delta_x
        y += delta_y
        theta += delta_theta
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        for car in PARKED_CARS:
            if circle_rect_collision(x, y, ROBOT_RADIUS, car):
                collision = True
                break
        if collision:
            running = False

     
        if not finish_crossed and prev_x < FINISH_X and x >= FINISH_X:
            finish_crossed = True
            running = False
        prev_x = x

   
        trail.append((x, y))

  
        cam_x += (x - cam_x) * 0.1
        cam_y += (y - cam_y) * 0.1


        screen.fill(BG_COLOR)

        def world_to_screen(wx, wy):
            sx = (wx - cam_x) * scale + WIDTH / 2
            sy = HEIGHT / 2 - (wy - cam_y) * scale
            return int(sx), int(sy)

  
        top_left = world_to_screen(WORLD_X_MIN, WORLD_Y_MAX)
        bottom_right = world_to_screen(WORLD_X_MAX, WORLD_Y_MIN)
        rect = pygame.Rect(top_left, (bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]))
        pygame.draw.rect(screen, WALL_COLOR, rect, 2)

        for car in PARKED_CARS:
            cx, cy, cw, ch = car
            p1 = world_to_screen(cx, cy + ch)
            p2 = world_to_screen(cx + cw, cy)
            car_rect = pygame.Rect(p1, (p2[0]-p1[0], p2[1]-p1[1]))
            pygame.draw.rect(screen, CAR_COLOR, car_rect)
            pygame.draw.rect(screen, (255,255,255), car_rect, 1)

      
        for yy in range(int(WORLD_Y_MIN*10), int(WORLD_Y_MAX*10), 1):
            wy = yy / 10.0
            p_start = world_to_screen(FINISH_X, wy)
            p_end = world_to_screen(FINISH_X, wy + 0.05)
            if int(yy) % 2 == 0:
                pygame.draw.line(screen, FINISH_COLOR, p_start, p_end, 2)

        if len(trail) > 1:
            points = [world_to_screen(px, py) for px, py in trail]
            pygame.draw.lines(screen, TRAIL_COLOR, False, points, 1)

       
        nose_x = x + 0.06 * math.cos(theta)
        nose_y = y + 0.06 * math.sin(theta)
        left_x  = x - 0.04 * math.cos(theta) + 0.04 * math.cos(theta + 2.5)
        left_y  = y - 0.04 * math.sin(theta) + 0.04 * math.sin(theta + 2.5)
        right_x = x - 0.04 * math.cos(theta) + 0.04 * math.cos(theta - 2.5)
        right_y = y - 0.04 * math.sin(theta) + 0.04 * math.sin(theta - 2.5)
        p_nose = world_to_screen(nose_x, nose_y)
        p_left = world_to_screen(left_x, left_y)
        p_right = world_to_screen(right_x, right_y)
        pygame.draw.polygon(screen, ROBOT_COLOR, [p_nose, p_left, p_right], 0)

     
        status_text = f"Comando: {command}"
        if command == 'stop':
            status_text += " (parado)"
        info_lines = [
            f"Vel: {current_speed:.2f} m/s",
            f"Pos: ({x:.2f}, {y:.2f})  θ: {math.degrees(theta):.0f}°",
            status_text,
            "ESC p/ sair"
        ]
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255,255,255))
            screen.blit(text, (10, 10 + i*20))

        if finish_crossed:
            msg = font.render("VITORIA! Cruzou a linha de chegada!", True, (255,255,0))
            screen.blit(msg, (WIDTH//2 - 150, HEIGHT//2))
        if collision:
            msg = font.render("COLISÃO! Fim da simulação.", True, (255,0,0))
            screen.blit(msg, (WIDTH//2 - 120, HEIGHT//2))

        pygame.display.flip()

        if finish_crossed or collision:
            pygame.time.wait(2000)
            running = False

    pygame.quit()
    return trail, finish_crossed, collision


def main():
    print("Simulador interativo com pista, obstáculos e linha de chegada.")
    print("Novos controles:")
    print("  ↑ : andar reto")
    print("  ← : curva à esquerda")
    print("  → : curva à direita")
    print("  ESPAÇO : parar")
    print("  ↓ : girar 180° (parar + meia‑volta)")
    print("  ESC : sair\n")
    r      = float(input("Raio da roda r (m) [0.034]: ") or 0.034)
    two_b  = float(input("Distância entre rodas 2b (m) [0.094]: ") or 0.094)
    b = two_b / 2.0
    T_ms   = float(input("Período de integração T (ms) [20]: ") or 20)
    dt = T_ms / 1000.0
    x0     = float(input("x inicial (m) [0.0]: ") or 0.0)
    y0     = float(input("y inicial (m) [0.0]: ") or 0.0)
    theta0 = float(input("θ inicial (rad) [0.0]: ") or 0.0)

    print(f"\nPista: x=[{WORLD_X_MIN},{WORLD_X_MAX}] m, y=[{WORLD_Y_MIN},{WORLD_Y_MAX}] m")
    print(f"Linha de chegada em x = {FINISH_X} m")
    print("Carros estacionados (azuis) – colisão encerra o programa.\n")
    input("Pressione Enter para iniciar...")

    trail, finished, crashed = interactive_simulation(r, b, dt, x0, y0, theta0)

    if finished:
        print("\n🏁 Parabéns! Você cruzou a linha de chegada!")
    elif crashed:
        print("\n💥 Você colidiu com um carro estacionado!")
    else:
        print("\nSimulação interrompida pelo usuário.")

    n = len(trail)
    step = max(1, n // 20)
    print("\nCaminho percorrido (amostras):")
    for i in range(0, n, step):
        px, py = trail[i]
        print(f"  ponto {i}: ({px:.4f}, {py:.4f})")
    if n > 0:
        px, py = trail[-1]
        print(f"  ponto {n-1}: ({px:.4f}, {py:.4f})")
    print(f"Total de pontos: {n}")

    if trail:
        traj_x, traj_y = zip(*trail)
        plt.figure(figsize=(6,6))
        plt.plot(traj_x, traj_y, '.-', markersize=2, linewidth=1)
        plt.plot(traj_x[0], traj_y[0], 'go', label='Início')
        plt.plot(traj_x[-1], traj_y[-1], 'ro', label='Fim')
        plt.axvline(WORLD_X_MIN, color='gray', linestyle='--')
        plt.axvline(WORLD_X_MAX, color='gray', linestyle='--')
        plt.axhline(WORLD_Y_MIN, color='gray', linestyle='--')
        plt.axhline(WORLD_Y_MAX, color='gray', linestyle='--')
        plt.axvline(FINISH_X, color='yellow', linestyle='-', linewidth=2, label='Chegada')
        for (cx, cy, cw, ch) in PARKED_CARS:
            rect = plt.Rectangle((cx, cy), cw, ch, color='blue', alpha=0.5)
            plt.gca().add_patch(rect)
        plt.axis('equal')
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Trajetória do robô na pista")
        plt.legend()
        plt.grid(True)
        plt.show()


main()