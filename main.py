from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, ClockObject, Vec3

loadPrcFileData("", "window-title Panda3D Car Simulation")
loadPrcFileData("", "win-size 800 600")

class CarSimulation(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.accept("escape", self.userExit)
        self.environ = self.loader.loadModel("models/environment")
        self.environ.reparentTo(self.render)
        self.environ.setScale(0.1)
        self.environ.setPos(-8, 42, 0)
        self.car = self.loader.loadModel("models/box")
        self.car.reparentTo(self.render)
        self.car.setScale(0.5, 1.0, 0.5)
        self.car.setPos(0, 0, 0.5)
        self.cameraOffset = Vec3(0, -10, 4)
        self.camera.reparentTo(self.render)
        self.camera.setPos(self.car.getPos() + self.car.getQuat().xform(self.cameraOffset))
        self.camera.lookAt(self.car)
        self.keyMap = {"forward": False, "backward": False, "left": False, "right": False}
        self.accept("w", self.setKey, ["forward", True])
        self.accept("w-up", self.setKey, ["forward", False])
        self.accept("s", self.setKey, ["backward", True])
        self.accept("s-up", self.setKey, ["backward", False])
        self.accept("a", self.setKey, ["left", True])
        self.accept("a-up", self.setKey, ["left", False])
        self.accept("d", self.setKey, ["right", True])
        self.accept("d-up", self.setKey, ["right", False])
        self.velocity = 0.0
        self.taskMgr.add(self.updateTask, "updateTask")

    def setKey(self, key, value):
        self.keyMap[key] = value

    def updateTask(self, task):
        dt = ClockObject.getGlobalClock().getDt()
        acceleration_rate = 30.0
        deceleration_rate = 40.0
        friction = 10.0
        max_speed = 20.0
        reverse_speed = max_speed / 3
        if self.keyMap["forward"]:
            self.velocity += acceleration_rate * dt
        elif self.keyMap["backward"]:
            self.velocity -= deceleration_rate * dt
        else:
            if self.velocity > 0:
                self.velocity -= friction * dt
                if self.velocity < 0:
                    self.velocity = 0
            elif self.velocity < 0:
                self.velocity += friction * dt
                if self.velocity > 0:
                    self.velocity = 0
        if self.velocity > max_speed:
            self.velocity = max_speed
        if self.velocity < -reverse_speed:
            self.velocity = -reverse_speed
        distance = self.velocity * dt
        self.car.setY(self.car, distance)
        turn_rate = 80.0
        if abs(self.velocity) > 0.1:
            turn = 0.0
            if self.keyMap["left"]:
                turn = turn_rate * dt * (abs(self.velocity) / max_speed)
            if self.keyMap["right"]:
                turn = -turn_rate * dt * (abs(self.velocity) / max_speed)
            self.car.setH(self.car.getH() + turn)
        desiredCamPos = self.car.getPos(self.render) + self.car.getQuat(self.render).xform(self.cameraOffset)
        smoothingFactor = 5.0
        currentCamPos = self.camera.getPos()
        newCamPos = currentCamPos + (desiredCamPos - currentCamPos) * (smoothingFactor * dt)
        self.camera.setPos(newCamPos)
        self.camera.lookAt(self.car)
        return task.cont

if __name__ == "__main__":
    app = CarSimulation()
    app.run()
