import torch


def B_batch(x, grid, k=0, extend=True, device='cpu'): #se calcula la B spline

    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k, device="cpu"):
 
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval


# def curve2coef(x_eval, y_eval, grid, k):

#     #print('haha', x_eval.shape, y_eval.shape, grid.shape)
#     batch = x_eval.shape[0]
#     in_dim = x_eval.shape[1]
#     out_dim = y_eval.shape[2]
#     n_coef = grid.shape[1] - k - 1
#     mat = B_batch(x_eval, grid, k)
#     mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
#     #print('mat', mat.shape)
#     y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
#     #print('y_eval', y_eval.shape)
#     device = mat.device
    
#     #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
#     try:
#         coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
#     except:
#         print('lstsq failed')

#     return coef

def curve2coef(x_eval, y_eval, grid, k):

    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1

    mat = B_batch(x_eval, grid, k)

    mat = (
        mat.permute(1, 0, 2)[:, None, :, :]
        .expand(in_dim, out_dim, batch, n_coef)
    )

    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)

    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]

    except Exception as e:

        print("\n========== LSTSQ ERROR ==========")
        print("Exception:", repr(e))

        print("\nShapes:")
        print("x_eval :", x_eval.shape)
        print("y_eval :", y_eval.shape)
        print("grid   :", grid.shape)
        print("mat    :", mat.shape)

        print("\nParameters:")
        print("k      :", k)
        print("batch  :", batch)
        print("in_dim :", in_dim)
        print("out_dim:", out_dim)
        print("n_coef :", n_coef)

        print("\nDtypes:")
        print("x_eval :", x_eval.dtype)
        print("y_eval :", y_eval.dtype)
        print("grid   :", grid.dtype)
        print("mat    :", mat.dtype)

        print("\nDevices:")
        print("x_eval :", x_eval.device)
        print("y_eval :", y_eval.device)
        print("grid   :", grid.device)
        print("mat    :", mat.device)

        print("\nFinite values:")
        print("x_eval finite :", torch.isfinite(x_eval).all().item())
        print("y_eval finite :", torch.isfinite(y_eval).all().item())
        print("grid finite   :", torch.isfinite(grid).all().item())
        print("mat finite    :", torch.isfinite(mat).all().item())

        print("\nRanges:")
        print(
            "x_eval:",
            x_eval.min().item(),
            x_eval.max().item()
        )
        print(
            "y_eval:",
            y_eval.min().item(),
            y_eval.max().item()
        )
        print(
            "grid:",
            grid.min().item(),
            grid.max().item()
        )
        print(
            "mat:",
            mat.min().item(),
            mat.max().item()
        )

        print("\nGrid differences:")
        grid_diff = grid[:, 1:] - grid[:, :-1]

        print(
            "min diff:",
            grid_diff.min().item()
        )

        print(
            "max diff:",
            grid_diff.max().item()
        )

        print(
            "zero diffs:",
            (grid_diff == 0).sum().item()
        )

        print(
            "negative diffs:",
            (grid_diff < 0).sum().item()
        )

        print("=================================\n")

        raise

    return coef


def extend_grid(grid, k_extend=0):

    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid