from __future__ import annotations
import math
import random
from typing import List, Tuple

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QWidget

# Panda3D imports
from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')
loadPrcFileData('', 'show-frame-rate-meter 0')
loadPrcFileData('', 'sync-video 1')
loadPrcFileData('', 'framebuffer-multisample 1')
loadPrcFileData('', 'multisamples 4')

from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, AntialiasAttrib, LineSegs, Vec3, Vec4, Point3, AmbientLight, DirectionalLight, ClockObject


class Panda3DWidget(QWidget):
    """Qt widget that embeds a Panda3D window and renders a modern 3D road scene."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        # Create Panda3D base without opening a window yet
        self.base = ShowBase(windowType='none')
        self.win = None
        self._window_opened = False

        # Scene/lifecycle state
        self._scene_built = False
        self.paused = False  # animation pause flag

        # Timer to step Panda3D inside Qt
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(16)  # ~60 FPS

        # Mouse interactions
        self._mouse_last = None
        self.setMouseTracking(True)

    def paintEvent(self, event):
        # Prevent Qt from drawing over the embedded Panda window
        return

    # ---------- Qt events ----------
    def showEvent(self, event):
        super().showEvent(event)
        if not self._window_opened:
            self._open_embedded_window()
            self._init_scene()
            self._window_opened = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, 'base', None) and getattr(self.base, 'win', None):
            props = WindowProperties()
            props.setSize(max(2, self.width()), max(2, self.height()))
            self.base.win.requestProperties(props)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    # ---------- Lifecycle ----------
    def shutdown(self):
        if self.timer.isActive():
            self.timer.stop()
        if getattr(self, 'base', None):
            try:
                self.base.userExit()
            except Exception:
                try:
                    self.base.destroy()
                except Exception:
                    pass
            self.base = None

    # ---------- Panda window embedding ----------
    def _open_embedded_window(self):
        handle = int(self.winId())
        props = WindowProperties()
        props.setOrigin(0, 0)
        props.setSize(max(2, self.width()), max(2, self.height()))
        props.setParentWindow(handle)
        self.win = self.base.openDefaultWindow(props=props)
        if self.win is None:
            raise RuntimeError('Failed to open embedded Panda3D window')

    def _step(self):
        if getattr(self, 'base', None):
            self.base.taskMgr.step()

    # ---------- Scene building ----------
    def _init_scene(self):
        if self._scene_built:
            return
        # Scene setup
        self.base.setBackgroundColor(0, 0, 0)  # Black background
        self.base.render.setAntialias(AntialiasAttrib.MAuto)

        # Lighting
        amb = AmbientLight('amb')
        amb.setColor(Vec4(0.15, 0.15, 0.18, 1))
        amb_np = self.base.render.attachNewNode(amb)
        self.base.render.setLight(amb_np)

        key = DirectionalLight('key')
        key.setColor(Vec4(0.9, 0.9, 1.0, 1) * 0.35)
        key_np = self.base.render.attachNewNode(key)
        key_np.setHpr(-45, -35, 0)
        self.base.render.setLight(key_np)

        # Camera
        self.base.disableMouse()
        self.cam_target = Point3(0, 0, 0)
        self.distance = 120.0
        self.azimuth = 45.0
        self.elevation = 35.0
        self._update_camera()

        # Build roads and route
        self.grid_size = 10  # number of blocks from center to edge per axis
        self.block = 12.0    # block size
        self.road_nodes: List = []
        self.route_nodes: List[Point3] = []

        self._build_roads()
        self._build_random_route()
        self._spawn_car()

        # Animate car along route
        self.car_speed = 18.0  # units per second
        self.route_progress = 0.0
        self.base.taskMgr.add(self._move_car_task, 'move-car')

        self._scene_built = True

    # ---------- Camera control ----------
    def _update_camera(self):
        # Convert spherical to cartesian
        rad_az = math.radians(self.azimuth)
        rad_el = math.radians(self.elevation)
        x = self.distance * math.cos(rad_el) * math.cos(rad_az)
        y = self.distance * math.cos(rad_el) * math.sin(rad_az)
        z = self.distance * math.sin(rad_el)
        self.base.camera.setPos(self.cam_target + Vec3(x, y, z))
        self.base.camera.lookAt(self.cam_target)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.distance = max(10.0, min(400.0, self.distance * (0.9 ** delta)))
        self._update_camera()

    def zoom_in(self):
        """Zoom camera in a step."""
        self.distance = max(10.0, self.distance * 0.9)
        self._update_camera()

    def zoom_out(self):
        """Zoom camera out a step."""
        self.distance = min(400.0, self.distance / 0.9)
        self._update_camera()

    def reset_view(self):
        """Reset camera orbit to defaults."""
        self.distance = 120.0
        self.azimuth = 45.0
        self.elevation = 35.0
        self.cam_target = Point3(0, 0, 0)
        self._update_camera()

    def center_on_car(self):
        """Center camera target on the car's current position (ground-projected)."""
        try:
            p = self.car.getPos(self.base.render)
            self.cam_target = Point3(p.x, p.y, 0)
            self._update_camera()
        except Exception:
            pass

    def toggle_pause(self):
        """Toggle pause of car movement along the route."""
        self.paused = not self.paused
        return self.paused

    def mousePressEvent(self, event):
        self._mouse_last = (event.position().x(), event.position().y(), event.buttons())

    def mouseMoveEvent(self, event):
        if self._mouse_last is None:
            return
        x, y, btns = self._mouse_last
        dx = event.position().x() - x
        dy = event.position().y() - y
        self._mouse_last = (event.position().x(), event.position().y(), event.buttons())

        if event.buttons() & Qt.MouseButton.RightButton:  # Right button: orbit
            self.azimuth = (self.azimuth - dx * 0.25) % 360.0
            self.elevation = max(5.0, min(85.0, self.elevation + dy * 0.25))
            self._update_camera()
        elif event.buttons() & Qt.MouseButton.MiddleButton:  # Middle button: pan
            pan_speed = self.distance * 0.002
            right = self.base.camera.getQuat(self.base.render).getRight()
            up = self.base.camera.getQuat(self.base.render).getUp()
            self.cam_target += (-right * dx + up * dy) * pan_speed
            self._update_camera()

    def mouseReleaseEvent(self, event):
        self._mouse_last = None

    def _build_roads(self):
        # Dark grey roads as grid on X-Y plane (Z=0)
        grid_extent = self.grid_size * self.block
        lines = LineSegs()
        lines.setColor(0.16, 0.16, 0.18, 1)
        lines.setThickness(4.0)

        for i in range(-self.grid_size, self.grid_size + 1):
            # Vertical streets (constant X)
            x = i * self.block
            lines.moveTo(x, -grid_extent, 0)
            lines.drawTo(x, grid_extent, 0)
            # Horizontal streets (constant Y)
            y = i * self.block
            lines.moveTo(-grid_extent, y, 0)
            lines.drawTo(grid_extent, y, 0)

        node = lines.create()
        self.road_np = self.base.render.attachNewNode(node)
        self.road_np.setTransparency(False)

        # Add subtle center markers for modern feel
        center = LineSegs()
        center.setColor(0.3, 0.3, 0.35, 1)
        center.setThickness(2.0)
        center.moveTo(-grid_extent, 0, 0.01)
        center.drawTo(grid_extent, 0, 0.01)
        center.moveTo(0, -grid_extent, 0.01)
        center.drawTo(0, grid_extent, 0.01)
        self.base.render.attachNewNode(center.create())

    def _build_random_route(self):
        # Create a Manhattan-style random route across the grid
        def grid_to_world(ix: int, iy: int) -> Point3:
            return Point3(ix * self.block, iy * self.block, 0.02)

        size = self.grid_size
        start = (random.randint(-size, -size//2), random.randint(-size, size))
        end = (random.randint(size//2, size), random.randint(-size, size))

        # Build a path that meanders from start to end
        path: List[Tuple[int, int]] = [start]
        cx, cy = start
        while (cx, cy) != end:
            if cx < end[0] and random.random() < 0.7:
                cx += 1
            elif cx > end[0] and random.random() < 0.7:
                cx -= 1
            else:
                cy += 1 if cy < end[1] else -1 if cy > end[1] else 0
            if (cx, cy) != path[-1]:
                path.append((cx, cy))

        # Convert to world points and draw blue line
        self.route_nodes = [grid_to_world(ix, iy) for ix, iy in path]

        route = LineSegs()
        route.setColor(0.1, 0.55, 1.0, 1)
        route.setThickness(6.0)
        for i, p in enumerate(self.route_nodes):
            if i == 0:
                route.moveTo(p)
            else:
                route.drawTo(p)
        self.base.render.attachNewNode(route.create())

    def _spawn_car(self):
        # Try to load a built-in cube; fallback to a simple card-made box
        try:
            car = self.base.loader.loadModel('models/misc/rgbCube')
            car.setScale(1.8, 3.2, 1.0)
        except Exception:
            from panda3d.core import CardMaker
            root = self.base.render.attachNewNode('car')
            # Create 6 faces using CardMaker
            def make_face(name, pos: Vec3, hpr: Vec3, sx: float, sy: float):
                cm = CardMaker(name)
                cm.setFrame(-sx/2, sx/2, -sy/2, sy/2)
                np = root.attachNewNode(cm.generate())
                np.setPos(pos)
                np.setHpr(hpr)
                return np
            sx, sy, sz = 1.8, 3.2, 1.0
            # Top and bottom
            make_face('top', Vec3(0, 0, sz/2), Vec3(0, 0, 0), sx, sy)
            make_face('bottom', Vec3(0, 0, -sz/2), Vec3(180, 0, 0), sx, sy)
            # Sides
            make_face('front', Vec3(0, sy/2, 0), Vec3(-90, 0, 0), sx, sz)
            make_face('back', Vec3(0, -sy/2, 0), Vec3(90, 0, 0), sx, sz)
            make_face('left', Vec3(-sx/2, 0), Vec3(0, 90, 0), sy, sz)
            make_face('right', Vec3(sx/2, 0), Vec3(0, -90, 0), sy, sz)
            car = root
        car.setColor(1.0, 1.0, 1.0, 1)
        car.setPos(self.route_nodes[0] if self.route_nodes else Point3(0, 0, 0.5))
        car.setZ(0.5)
        # Keep NodePath and then reparent (reparentTo returns None)
        self.car = car
        self.car.reparentTo(self.base.render)

        # Simple shadow/ground glow
        shadow = LineSegs()
        shadow.setColor(0.0, 0.0, 0.0, 0.5)
        shadow.setThickness(14.0)
        p = self.car.getPos()
        shadow.moveTo(p.x - 0.4, p.y, 0.015)
        shadow.drawTo(p.x + 0.4, p.y, 0.015)
        self.base.render.attachNewNode(shadow.create())

    # ---------- Animation ----------
    def _move_car_task(self, task):
        if not self.route_nodes:
            return task.cont

        # Compute total route length
        total_len = 0.0
        seg_lengths: List[float] = []
        for i in range(1, len(self.route_nodes)):
            a = self.route_nodes[i-1]
            b = self.route_nodes[i]
            l = (b - a).length()
            seg_lengths.append(l)
            total_len += l

        if total_len <= 1e-3:
            return task.cont

        # Advance progress
        dt = ClockObject.getGlobalClock().getDt()
        if not self.paused:
            self.route_progress = (self.route_progress + self.car_speed * dt) % total_len

        # Find segment and interpolate
        d = self.route_progress
        seg_index = 0
        while d > seg_lengths[seg_index]:
            d -= seg_lengths[seg_index]
            seg_index = (seg_index + 1) % len(seg_lengths)
        a = self.route_nodes[seg_index]
        b = self.route_nodes[(seg_index + 1) % len(self.route_nodes)]
        t = d / max(1e-6, seg_lengths[seg_index])
        pos = a * (1 - t) + b * t
        self.car.setPos(pos.x, pos.y, 0.5)

        # Orient car along path
        dir_vec = (b - a)
        heading = math.degrees(math.atan2(dir_vec.y, dir_vec.x)) - 90.0
        self.car.setHpr(heading, 0, 0)

        return task.cont
