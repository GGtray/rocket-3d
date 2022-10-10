import numpy as np
import utils
import cv2

world_x_min = -300  # meters
world_x_max = 300
world_y_min = -30
world_y_max = 570

H = 50
g =9.8
dt = 0.05

path_to_bg_img = "./bg_img.jpg"
viewport_h = 768
viewport_h = int(viewport_h)
viewport_w = int(viewport_h * (world_x_max-world_x_min) \
                          / (world_y_max - world_y_min))
target_r = 50

def render(window_name='new_world', wait_time=1,
            with_trajectory=True, with_camera_tracking=True,
            crop_scale=0.4):

    
    bg_img = utils.load_bg_img(path_to_bg_img, w=viewport_w, h=viewport_h)


    canvas = np.copy(bg_img)
    polys = create_polygons()

    # draw target region
    for poly in polys['target_region']:
        draw_a_polygon(canvas, poly)
    # draw rocket
    for poly in polys['rocket']:
        draw_a_polygon(canvas, poly)
    frame_0 = canvas.copy()

    # draw engine work
    for poly in polys['engine_work']:
        self.draw_a_polygon(canvas, poly)
    frame_1 = canvas.copy()

    if with_camera_tracking:
        frame_0 = crop_alongwith_camera(frame_0, crop_scale=crop_scale)
        frame_1 = crop_alongwith_camera(frame_1, crop_scale=crop_scale)

    # draw trajectory
    if with_trajectory:
        draw_trajectory(frame_0)
        draw_trajectory(frame_1)

    # draw text
    draw_text(frame_0, color=(0, 0, 0))
    draw_text(frame_1, color=(0, 0, 0))

    cv2.imshow(window_name, frame_0[:,:,::-1])
    cv2.waitKey(wait_time)
    cv2.imshow(window_name, frame_1[:,:,::-1])
    cv2.waitKey(wait_time)
    return frame_0, frame_1

def create_polygons(state):

    polys = {'rocket': [], 'engine_work': [], 'target_region': []}

    H, W = H, H/10
    dl = H / 30

    # rocket main body
    pts = [[-W/2, H/2], [W/2, H/2], [W/2, -H/2], [-W/2, -H/2]]
    polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})
    # rocket paint
    pts = utils.create_rectangle_poly(center=(0, -0.35*H), w=W, h=0.1*H)
    polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
    pts = utils.create_rectangle_poly(center=(0, -0.46*H), w=W, h=0.02*H)
    polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
    # rocket landing rack
    pts = [[-W/2, -H/2], [-W/2-H/10, -H/2-H/20], [-W/2, -H/2+H/20]]
    polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
    pts = [[W/2, -H/2], [W/2+H/10, -H/2-H/20], [W/2, -H/2+H/20]]
    polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})

    

    # engine work
    f, phi = state['f'], state['phi']
    c, s = np.cos(phi), np.sin(phi)

    if f > 0 and f < 0.5 * g:
        pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
        pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
        polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
        polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
    elif f > 0.5 * g and f < 1.5 * g:
        pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
        pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
        pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
        polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
        polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
        polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
    elif f > 1.5 * g:
        pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
        pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
        pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
        pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
        polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
        polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
        polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
        polys['engine_work'].append({'pts': pts4, 'face_color': (255, 255, 255), 'edge_color': None})
    # target region
    
    pts1 = utils.create_ellipse_poly(center=(0, 0), rx=target_r, ry=target_r/4.0)
    pts2 = utils.create_rectangle_poly(center=(0, 0), w=target_r/3.0, h=0)
    pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=target_r/6.0)
    polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
    polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
    polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

    # apply transformation
    for poly in polys['rocket'] + polys['engine_work']:
        M = utils.create_pose_matrix(tx=state['x'], ty=state['y'], rz=state['theta'])
        pts = np.array(poly['pts'])
        pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
        pts = np.matmul(M, pts.T).T
        poly['pts'] = pts[:, 0:2]

    return polys


def draw_a_polygon(canvas, poly):

    pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
    pts_px = wd2pxl(pts)
    if face_color is not None:
        cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
    if edge_color is not None:
        cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

    return canvas


def wd2pxl(pts, to_int=True):

    pts_px = np.zeros_like(pts)

    scale = viewport_w / (world_x_max - world_x_min)
    for i in range(len(pts)):
        pt = pts[i]
        x_p = (pt[0] - world_x_min) * scale
        y_p = (pt[1] - world_y_min) * scale
        y_p = viewport_h - y_p
        pts_px[i] = [x_p, y_p]

    if to_int:
        return pts_px.astype(int)
    else:
        return pts_px

def draw_text(state, step_id, canvas, color=(255, 255, 0)):

    def put_text(vis, text, pt):
        cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

    pt = (10, 20)
    text = "simulation time: %.2fs" % (step_id * dt)
    put_text(canvas, text, pt)

    pt = (10, 40)
    text = "simulation steps: %d" % (step_id)
    put_text(canvas, text, pt)

    pt = (10, 60)
    text = "x: %.2f m, y: %.2f m" % \
            (state['x'], state['y'])
    put_text(canvas, text, pt)

    pt = (10, 80)
    text = "vx: %.2f m/s, vy: %.2f m/s" % \
            (state['vx'], state['vy'])
    put_text(canvas, text, pt)

    pt = (10, 100)
    text = "a: %.2f degree, va: %.2f degree/s" % \
            (state['theta'] * 180 / np.pi, state['vtheta'] * 180 / np.pi)
    put_text(canvas, text, pt)


def draw_trajectory(state_buffer, canvas, color=(255, 0, 0)):

    pannel_w, pannel_h = 256, 256
    traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

    sw, sh = pannel_w/viewport_w, pannel_h/viewport_h  # scale factors

    # draw horizon line
    range_x, range_y =world_x_max - world_x_min, world_y_max - world_y_min
    pts = [[world_x_min + range_x/3, H/2], [world_x_max - range_x/3, H/2]]
    pts_px = wd2pxl(pts)
    x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
    x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
    cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # draw vertical line
    pts = [[0, H/2], [0, H/2+range_y/20]]
    pts_px = wd2pxl(pts)
    x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
    x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
    cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    if len(state_buffer) < 2:
        return

    # draw traj
    pts = []
    for state in state_buffer:
        pts.append([state['x'], state['y']])
    pts_px = wd2pxl(pts)

    dn = 5
    for i in range(0, len(pts_px)-dn, dn):

        x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
        x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

    roi_x1, roi_x2 = viewport_w - 10 - pannel_w, viewport_w - 10
    roi_y1, roi_y2 = 10, 10 + pannel_h
    canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel



def crop_alongwith_camera(state, vis, crop_scale=0.4):
    x, y = state['x'], state['y']
    xp, yp = wd2pxl([[x, y]])[0]
    crop_w_half, crop_h_half = int(viewport_w*crop_scale), int(viewport_h*crop_scale)
    # check boundary
    if xp <= crop_w_half + 1:
        xp = crop_w_half + 1
    if xp >= viewport_w - crop_w_half - 1:
        xp = viewport_w - crop_w_half - 1
    if yp <= crop_h_half + 1:
        yp = crop_h_half + 1
    if yp >= viewport_h - crop_h_half - 1:
        yp = viewport_h - crop_h_half - 1

    x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
    vis = vis[y1:y2, x1:x2, :]

    vis = cv2.resize(vis, (viewport_w, viewport_h))
    return vis