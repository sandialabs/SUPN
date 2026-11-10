from supn import SUPN
import torch
torch.set_default_dtype(torch.float64)


def test_initialization():
    net = SUPN(max_degree=3, width=10, d=1)
    nparams = sum([param.numel() for param in net.parameters()])
    assert nparams == 50, 'Error in 1D initialization'

    net = SUPN(max_degree=3, width=10, d=2, space='total_degree')
    nparams = sum([param.numel() for param in net.parameters()])
    assert nparams == 110, 'Error in 2D initialization'


def test_differentiation_1d():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 10001)
    h = x[1] - x[0]
    net = SUPN(max_degree=3, width=10, d=1)
    derivatives = net.dx(x[:, None], order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    y = net(x[:, None]).flatten()

    du_approx = (y[2:] - y[:-2])/(2*h)
    du_err = ((du_approx - derivatives[0, 1:-1]).norm() /
              derivatives[0, 1:-1].norm())
    assert du_err < 4e-6, 'Error in 1D 1st derivative'

    d2_approx = (y[2:] - 2*y[1:-1] + y[:-2])/(h**2)
    d2u_err = ((d2_approx - derivatives[1, 1:-1]).norm() /
               derivatives[1, 1:-1].norm())
    assert d2u_err < 6e-6, 'Error in 1D 2nd derivative'

    d3u_approx = (y[4:] - 2*y[3:-1] + 2*y[1:-3] - y[:-4])/(2 * h**3)
    d3u_err = ((d3u_approx - derivatives[2, 2:-2]).norm() /
               derivatives[2, 2:-2].norm())
    assert d3u_err < 4e-5, 'Error in 1D 3rd derivative'

    d4u_approx = (y[4:] - 4*y[3:-1] + 6*y[2:-2] - 4*y[1:-3] + y[:-4]) / (h**4)
    d4u_err = ((d4u_approx - derivatives[3, 2:-2]).norm() /
               derivatives[3, 2:-2].norm())
    assert d4u_err < 5e-5, 'Error in 1D 4th derivative'


def test_differentiation_2d():
    torch.manual_seed(0)

    # Prepare for autodiff
    x_1d = torch.linspace(-1, 3, 21, requires_grad=True)
    y_1d = torch.linspace(-1, 1, 21, requires_grad=True)
    z_1d = torch.linspace(2, 7, 3, requires_grad=True)  # for sanity check
    (X, Y, Z) = torch.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x, y, z = X.flatten(), Y.flatten(), Z.flatten()
    grid = torch.stack((x, y, z), dim=-1)
    domain_transform = torch.as_tensor([[-1, 3], [-1, 1], [2, 7]])
    net = SUPN(max_degree=6, width=10, d=3, space='total_degree',
               domain_transform=domain_transform)
    u = net(grid).flatten()
    du, du_xxyy = net.dx(grid, order=[1, 2, 3, 4], return_xxyy=True)
    _, _ = net.dx(grid, order=4, return_xxyy=True)

    # Compute autodiff
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx.sum(), x, create_graph=True)[0]

    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_yyy = torch.autograd.grad(u_yy.sum(), y, create_graph=True)[0]
    u_yyyy = torch.autograd.grad(u_yyy.sum(), y, create_graph=True)[0]

    u_xxy = torch.autograd.grad(u_xx.sum(), y, create_graph=True)[0]
    u_xxyy = torch.autograd.grad(u_xxy.sum(), y, create_graph=True)[0]

    u_x_err = ((du[0, 0, ...].flatten() - u_x.flatten()).norm() /
               u_x.norm()).item()
    u_y_err = ((du[0, 1, ...].flatten() - u_y.flatten()).norm() /
               u_y.norm()).item()
    assert max(u_x_err, u_y_err) < 1e-14, 'Error in 2D 1st derivative'

    u_xx_err = ((du[1, 0, ...].flatten() - u_xx.flatten()).norm() /
                u_xx.norm()).item()
    u_yy_err = ((du[1, 1, ...].flatten() - u_yy.flatten()).norm() /
                u_y.norm()).item()
    assert max(u_xx_err, u_yy_err) < 1e-14, 'Error in 2D 2nd derivative'

    u_xxx_err = ((du[2, 0, ...].flatten() - u_xxx.flatten()).norm() /
                 u_xxx.norm()).item()
    u_yyy_err = ((du[2, 1, ...].flatten() - u_yyy.flatten()).norm() /
                 u_yyy.norm()).item()
    assert max(u_xxx_err, u_yyy_err) < 1e-14, 'Error in 2D 3rd derivative'

    u_xxxx_err = ((du[3, 0, ...].flatten() - u_xxxx.flatten()).norm() /
                  u_xxxx.norm()).item()
    u_yyyy_err = ((du[3, 1, ...].flatten() - u_yyyy.flatten()).norm() /
                  u_yyyy.norm()).item()
    assert max(u_xxxx_err, u_yyyy_err) < 1e-14, 'Error in 2D 4rd derivative'

    u_xxyy_err = ((u_xxyy.flatten() - du_xxyy.flatten()).norm()
                  / u_xxyy.norm()).item()
    assert u_xxyy_err < 1e-14, f'Error in u_xxyy: {u_xxyy_err}'


def test_precompute_data():
    net = SUPN(max_degree=5, width=1, d=1)
    x = torch.linspace(-1, 1, 101)[:, None]
    net.precompute_data(x)
    cheby_mat = torch.cos(torch.arange(6)[None, :] * torch.arccos(x))
    diff = ((cheby_mat.flatten() -
             net.precomputed_chebyshev_matrix.flatten()).norm())
    assert diff < 1e-13, 'Error in three-term recurrence'

    net = SUPN(max_degree=5, width=1, d=1, x_train=x)
    net.precompute_data(x)
    assert (net(x) == net(x, recompute=True)).all(), 'Error precomputing'

    assert (net._three_term_recurrence(x.flatten(), 0) == 1).all()


def test_domain_transform():
    domain_transform = torch.as_tensor([[2, 7]])
    net = SUPN(max_degree=5, width=1, d=1, domain_transform=domain_transform)
    x = torch.linspace(2, 7, 101)[:, None]
    diff = (net._transform(x).flatten() - torch.linspace(-1, 1, 101)).norm()
    assert diff < 1e-14, 'Error in 1D domain transform'

    domain_transform = torch.as_tensor([[2, 7], [-2, 1]])
    net = SUPN(max_degree=5, width=1, d=2, domain_transform=domain_transform)
    x = torch.linspace(2, 7, 101)
    y = torch.linspace(-2, 1, 51)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    x_c = torch.linspace(-1, 1, 101)
    y_c = torch.linspace(-1, 1, 51)
    (X_c, Y_c) = torch.meshgrid(x_c, y_c, indexing='ij')

    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    grid_c = torch.stack((X_c.flatten(), Y_c.flatten()), dim=-1)
    diff = (net._transform(grid).flatten() - grid_c.flatten()).norm()
    assert diff < 1e-13, 'Error in 2D domain transform'

    x = torch.linspace(-1, 1, 51)[:, None]
    net = SUPN(max_degree=3, width=10, d=1)
    derivatives = net.dx(x, order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    net.set_domain_transform(torch.as_tensor([[3, 7]]))
    x = torch.linspace(3, 7, 51)[:, None]
    new_derivatives = net.dx(x, order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    adjustment = 2 ** torch.arange(1, 5)[:, None]
    err = ((new_derivatives * adjustment - derivatives).flatten().norm() /
           derivatives.flatten().norm())
    assert err < 1e-13, 'Error in differentiating affine transform (1D)'

    net.precompute_data(x)
    _derivatives = net.dx(order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    print(new_derivatives, _derivatives)
    assert ((_derivatives-new_derivatives).norm() / new_derivatives.norm()
            < 1e-13), 'Error in precomputation'

    x, y = torch.linspace(-1, 1, 51), torch.linspace(-1, 1, 51)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    net = SUPN(max_degree=3, width=10, d=2, space='total_degree')
    derivatives = net.dx(grid, order=[1, 2, 3, 4]).reshape(4, grid.shape[1],
                                                           *X.shape)
    net.set_domain_transform(torch.as_tensor([[3, 7], [0, 1]]))
    x, y = torch.linspace(3, 7, 51), torch.linspace(0, 1, 51)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    new_derivatives = net.dx(grid, order=[1, 2, 3, 4])
    new_derivatives = new_derivatives.reshape(4, grid.shape[1], *X.shape)
    adjustment = torch.as_tensor([[2], [0.5]]) ** torch.arange(1, 5)[None, :]
    adjustment = adjustment.T[..., None, None]
    err = ((adjustment * new_derivatives - derivatives).flatten().norm() /
           derivatives.flatten().norm())
    assert err < 1e-13, 'Error in differentiating affine transform (2D)'


def test_exception_handling():
    x = torch.linspace(-1, 1, 101)[:, None]
    net = SUPN(max_degree=5, width=1)

    # Incorrect transform
    try:
        net.set_domain_transform(torch.as_tensor([[1], [2]]))
    except ValueError:
        pass

    # Calling dx without evaluation point or precomputed data
    try:
        net.dx(order=1)
    except RuntimeError:
        pass

    # Derivatives over order 4
    try:
        net.dx(x, order=5)
    except ValueError:
        pass

    # Three-term recurrence with 2D input
    try:
        net._three_term_recurrence(x, 0)
    except ValueError:
        pass

    # Query point outside domain
    try:
        net._transform(2*x)
    except ValueError:
        pass
