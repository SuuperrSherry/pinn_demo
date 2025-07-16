import torch
from config import Config
from pinn import PINN_ArcLength
from loss import compute_loss

def train_segment(s_start, s_end, x_start, p_start, physics_fn,
                  num_points=Config.POINTS_PER_SEGMENT, max_epochs=Config.MAX_EPOCHS):
    model = PINN_ArcLength()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    s_train = torch.linspace(s_start, s_end, num_points).view(-1, 1)
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        total_loss, loss_physics, loss_arclength, loss_ic = compute_loss(
            model, s_train, x_start, p_start, s_start, physics_fn)
        total_loss.backward()
        optimizer.step()
        
        if epoch % Config.PRINT_INTERVAL == 0:
            print(f"Epoch {epoch:4d}, Total: {total_loss.item():.6f}, "
                  f"Phys: {loss_physics.item():.6f}, Arc: {loss_arclength.item():.6f}, IC: {loss_ic.item():.6f}")
    
    return model


def path_following(start_point, physics_fn,
                   total_length=Config.TOTAL_LENGTH, base_step=Config.BASE_STEP_SIZE,
                   max_segments=Config.MAX_SEGMENTS):
    x_current, p_current, s_current = start_point[0], start_point[1], 0.0
    path_x, path_p, path_s = [x_current], [p_current], [s_current]
    print(f"Starting path from ({x_current:.3f}, {p_current:.3f})")
    for segment in range(max_segments):
        if s_current >= total_length:
            break
        s_end = min(s_current + base_step, total_length)
        model = train_segment(s_current, s_end, 
                              torch.tensor([[x_current]]), torch.tensor([[p_current]]),
                              physics_fn)
        s_segment = torch.linspace(s_current, s_end, 50).view(-1, 1)
        with torch.no_grad():
            x_segment, p_segment = model(s_segment)
            s0 = torch.tensor([[s_current]], dtype=torch.float32)
            x0_pred, p0_pred = model(s0)
            print(f"Pred start: ({x0_pred.item():.4f}, {p0_pred.item():.4f}), Target: ({x_current:.4f}, {p_current:.4f})")
        path_x.extend(x_segment[1:].squeeze().tolist())
        path_p.extend(p_segment[1:].squeeze().tolist())
        path_s.extend(s_segment[1:].squeeze().tolist())
        x_current, p_current, s_current = x_segment[-1].item(), p_segment[-1].item(), s_end
        curvature_factor = 1.0 / (1.0 + abs(p_current))
        base_step = max(Config.MIN_STEP_SIZE, min(Config.MAX_STEP_SIZE, Config.BASE_STEP_SIZE * curvature_factor))
    return path_x, path_p, path_s