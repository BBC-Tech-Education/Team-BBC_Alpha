from controller import Robot
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from enum import Enum, auto
import math
import cv2
import struct

#This program can:
#It can detect full walls and store them
#It can detect half walls but can't store them
#It can detect the floor colours (tile.floor) and save them
#It can follow left wall and avoid black holes and swamps
#It can detect the direction robot is facing
#It can identify the presents of tokens, but it can't detect whether it is stable, harmed and unharmed
#It can know where it is in the maze
#It can receive messages

#Things to do:
#Create and submit the map
#detect half walls and store them
#identify and save loction of victims in the map
#detect curve walls and store them
#identify and save hazmets
#
#----------------------START OF CONSTANTS-------------------------

TIME_STEP = 32           # Set the time step for the simulation
MAX_VELOCITY = 6.28      # Set maximum velocity radians/s

DRIVE_SPEED = 3.0           # radians/s
TURN_SPEED = 1.0            # radians/s

WALL_TOLERANCE = 0.1      # Distance to test if a wall is close 0.1 m away
FRONT_WALL_TOLERANCE = 0.05
DEBUG = True            # True when developing code - used to print out values

TILE_SIZE = 0.12            # Tile size in metres
TRAVEL_TOLERENCE = 0.009
YAW_TOLERENCE = math.radians(1.0)           # use a function process in the maths

# Directions
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

DIRECTION_NAMES = {
    NORTH: "NORTH",
    EAST: "EAST",
    SOUTH: "SOUTH",
    WEST: "WEST"
}
#---------------------END OF CONSTANTS-----------------------------

#---------------------START OF VARIABLES---------------------------
@dataclass
class Maze_data:
    start_position: Optional[Tuple[float, float, float]] = None    # maze starting position
    start_position_grid: Optional[Tuple[int, int]] = None 
    left_wall: bool = False
    front_wall: bool = False
    right_wall: bool = False
    left_hole: bool = False
 
    # Information about current position in the maze
    heading: float = 0.0                                            # current heading
    current_start_position: Optional[Tuple[float, float, float]] = None     # coordinates of current tile start position
    target_heading: float = 0.0                                     # target heading for the movement
    state_initialised: bool = False                                 # track the first time we enter a move state
    current_start_position_grid: Optional[Tuple[int, int]] = None 
    victim_found: bool = False 
    current_direction: int = NORTH
    swamp_mode: bool = False                                        # Track swamp status

class Tile_type(Enum):
    START = auto()
    BLUE = auto()
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    ORANGE = auto()
    PURPLE = auto()
    CHECK_POINTS = auto()
    BLACK_HOLE = auto()
    SWAMP = auto()
    NONE = auto()

class token_type(Enum):
    H = auto()
    S = auto()
    U = auto()
    F = auto()
    P = auto()
    C = auto()
    O = auto()
    NONE = auto()
    FAKE = auto()

@dataclass
class Maze_Tile:
    north_wall: bool = False
    east_wall: bool = False
    south_wall: bool = False
    west_wall: bool = False
    visited: int = 0
    floor: Tile_type = Tile_type.NONE
    victim: token_type = token_type.NONE

class Robot_States(Enum):
    EXPLORE = auto()
    DECIDE = auto()
    TURN_LEFT = auto()
    MOVE_FORWARD = auto()
    TURN_RIGHT = auto()
    SPIN = auto()
    BACK_TO_CENTRE = auto()

robot = Robot()

maze = Maze_data()

maze_map = {}

state = Robot_States
#---------------------END OF VARIABLES-------------------------


#-------------------START ROBOT OBJECTS ------------------

# Motors
l_motor = robot.getDevice("lw motor")
r_motor = robot.getDevice("rw motor")

# Initialise motors
l_motor.setPosition(float("inf"))
r_motor.setPosition(float("inf"))

l_motor.setVelocity(0)
r_motor.setVelocity(0)


# Distance sensors
l_ds_b = robot.getDevice("l_ds_b")
r_ds_b = robot.getDevice("r_ds_b")
l_ds_f = robot.getDevice("l_ds_f")
f_ds_r = robot.getDevice("f_ds_r")
r_ds_f = robot.getDevice("r_ds_f")
f_ds_m = robot.getDevice("f_ds_m")
f_ds_l = robot.getDevice("f_ds_l")

# Initialise distance sensors
l_ds_b.enable(TIME_STEP)
r_ds_b.enable(TIME_STEP)
l_ds_f.enable(TIME_STEP)
f_ds_r.enable(TIME_STEP)
r_ds_f.enable(TIME_STEP)
f_ds_m.enable(TIME_STEP)
f_ds_l.enable(TIME_STEP)

# GPS
gps = robot.getDevice("gps")  
gps.enable(TIME_STEP)

# Gyro
gyro = robot.getDevice("gyro")
gyro.enable(TIME_STEP)

# IMU
imu = robot.getDevice("inertial_unit")
imu.enable(TIME_STEP)

# Colour sensor
cs = robot.getDevice("colour")
cs.enable(TIME_STEP)

# Camera
cam = robot.getDevice("camera")
cam.enable(TIME_STEP)

emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(TIME_STEP)
#-------------------END ROBOT OBJECTS ------------------

#-------------------START OF FUNCTIONS--------------------

def print_info():
    print(
        f"Left_F: {l_ds_f.getValue():.3f} Left_B: {l_ds_b.getValue():.3f} LeftWall: {left_wall()}\n"
        f"Front_R: {f_ds_r.getValue():.3f} Front_M: {f_ds_m.getValue():.3f} Front_L: {f_ds_l.getValue():.3f} FrontWall: {front_wall()}\n"
        f"Right_F: {r_ds_f.getValue():.3f} Right_B: {r_ds_b.getValue():.3f} RightWall: {right_wall()}\n"
        f"Maze Pos: {maze.start_position} Grid: {maze.current_start_position_grid}"
    )

    roll, pitch, yaw = imu.getRollPitchYaw()

    print(
        f"Yaw: {yaw:.3f} "
        f"GPS: {[f'{x:.4f}' for x in gps.getValues()]}"
    )

def dump_map():
    print("\n~~~~~~~~~~~ MAZE MAP ~~~~~~~~~~~~~~~")
    print(
        f"Robot at: {maze.current_start_position_grid}  "
        f"Direction: {maze.current_direction}"
    )
    print(f"Tiles discovered: {len(maze_map)}")
    print()

    for coords in sorted(maze_map.keys()):
        tile = maze_map[coords]
        marker = ""

        if coords == maze.current_start_position_grid:
            marker += " <ROBOT>"
        if tile.floor == Tile_type.BLACK_HOLE:
            marker += " HOLE"
        if tile.floor == Tile_type.SWAMP:
            marker += " SWAMP"
        if tile.floor == Tile_type.START:
            marker += " START"
        if tile.floor == Tile_type.CHECK_POINTS:
            marker += " CHECK POINT"
        if tile.floor == Tile_type.BLUE:
            marker += " BLUE"
        if tile.floor == Tile_type.RED:
            marker += " RED"
        if tile.floor == Tile_type.YELLOW:
            marker += " YELLOW"
        if tile.floor == Tile_type.GREEN:
            marker += " GREEN"
        if tile.floor == Tile_type.ORANGE:
            marker += " ORANGE"
        if tile.floor == Tile_type.PURPLE:
            marker += " PURPLE"
        if tile.floor == Tile_type.NONE:
            marker += " NONE"
        print(
            f"{coords}{marker}"
        )

        print(
            f" V:{tile.visited} "
            f"N:{tile.north_wall} "
            f"E:{tile.east_wall} "
            f"S:{tile.south_wall} "         
            f"W:{tile.west_wall}"
        )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

def delay(ms):
    initTime = robot.getTime()
    while robot.step(TIME_STEP) != -1:
        if (robot.getTime() - initTime) * 1000.0 > ms:
            break

def drive_forward():
    l_motor.setVelocity(DRIVE_SPEED)
    r_motor.setVelocity(DRIVE_SPEED)

def drive_backward():
    l_motor.setVelocity(-DRIVE_SPEED)
    r_motor.setVelocity(-DRIVE_SPEED)

def stop():
    l_motor.setVelocity(0)
    r_motor.setVelocity(0)

def rotate_left():
    l_motor.setVelocity(-TURN_SPEED)
    r_motor.setVelocity(TURN_SPEED)

def rotate_right():
    l_motor.setVelocity(TURN_SPEED)
    r_motor.setVelocity(-TURN_SPEED)

def left_wall():
    return min(l_ds_f.getValue(), l_ds_b.getValue()) < WALL_TOLERANCE

def right_wall():
    return min(r_ds_f.getValue(), r_ds_b.getValue()) < WALL_TOLERANCE

def front_wall():
    return min(f_ds_r.getValue(), f_ds_m.getValue(), f_ds_l.getValue()) < WALL_TOLERANCE

def gps_data():
    return gps.getValues()

def distance_travelled(start, current):
    dist_x = current[0] - start[0]
    dist_z = current[2] - start[2]
    return np.sqrt(dist_x*dist_x + dist_z*dist_z)

def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def yaw_difference(current_yaw, target_yaw):
    raw_diff = target_yaw - current_yaw
    wrapped_diff = wrap_angle(raw_diff)
    shortest_diff = np.abs(wrapped_diff)
    return float(shortest_diff)

def get_heading():
    roll, pitch, yaw = imu.getRollPitchYaw()
    return yaw

def snap_to_cardinal(theta):
    return np.round(theta/(np.pi/2))*(np.pi/2)

def check_messages(state):
    if receiver.getQueueLength() > 0:
        rec_message = receiver.getBytes()

        if len(rec_message) == 1:
            tup = struct.unpack('c', rec_message)
            if tup[0].decode() == 'L':    # Lack of progress
                if DEBUG: print("Detected Lack of Progress!")
                stop()
                maze.state_initialised = False
                state = Robot_States.EXPLORE
            receiver.nextPacket()
        elif len(rec_message) == 16:
            tup = struct.unpack('c f i i', rec_message)
            if tup[0].decode() == 'G':    # Game Info
                print(f't={robot.getTime()}s SCORE= {tup[1]} Remaining time: {tup[3]}s ')                
            receiver.nextPacket()

    return state

def detect_victim():
    image_data = cam.getImage()
    coords_list = []
    img = np.array(np.frombuffer(image_data, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4)))
    img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 144, 255, cv2.THRESH_BINARY)[1]

    contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 300:
            coords = list(c[0][0])
            coords_list.append(coords)
            print("Victim at x="+str(coords[0])+" y="+str(coords[1])+" area="+str(cv2.contourArea(c)))
            if cv2.contourArea(c) > 500:
                maze.victim_found = True
            if cv2.contourArea(c) > 10000:
                cam.saveImage("victim.png", 100)

    return coords_list

def victim_report():
    (x, y, z) = gps_data()
    emitter.send(struct.pack('i i c', int(100*x), int(100*z), 'U'.encode())) 
    maze.victim_found = False

def determine_direction():
    yaw = get_heading()
    if DEBUG: print(f"current heading {int(-round(yaw / (np.pi / 2))) % 4}")
    return int(-round(yaw / (np.pi / 2))) % 4

def get_current_tile():
    (x, z) = maze.current_start_position_grid

    if (x, z) not in maze_map:
        maze_map[(x, z)] = Maze_Tile()
    return maze_map[(x, z)]

def get_floor_color_rgb():
    img = cs.getImage()
    if img is None:
        return 0, 0, 0
   
    r = cs.imageGetRed(img, 1, 0, 0)
    g = cs.imageGetGreen(img, 1, 0, 0)
    b = cs.imageGetBlue(img, 1, 0, 0)
   
    return r, g, b

def classify_floor_colour():
    r, g, b = get_floor_color_rgb()
   
    if r > 250 and g > 250 and b > 250:
        if DEBUG: print('CHECKPOINT (Silver)')
        return Tile_type.CHECK_POINTS
    elif r < 30 and g < 30 and b < 30:
        if DEBUG: print('BLACK HOLE')
        return Tile_type.BLACK_HOLE
    elif 140 <= r <= 190 and 100 <= g <= 150 and b < 80:
        if DEBUG: print('SWAMP')
        return Tile_type.SWAMP
    elif b > 200 and r < 100 and g < 100:
        if DEBUG: print('BLUE')
        return Tile_type.BLUE 
    elif g > 200 and r < 100 and b < 100:
        if DEBUG: print('Green')
        return Tile_type.GREEN  
    elif r > 200 and g < 100 and b < 100:
        if DEBUG: print('Red')
        return Tile_type.RED    
    elif r > 150 and b > 200 and g < 100:
        if DEBUG: print('Purple')
        return Tile_type.PURPLE  
    elif r > 200 and g > 245 and b < 100:
        if DEBUG: print('Yellow')
        return Tile_type.YELLOW
    elif r > 200 and 200 < g <= 245 and b < 100:
        if DEBUG: print('Orange')
        return Tile_type.ORANGE  
    else:
        return Tile_type.NONE

def update_tile():
    tile = get_current_tile()
    first_visit = tile.visited
    tile.visited += 1

    if not tile.floor == Tile_type.START: 
        tile.floor = classify_floor_colour()

    if tile.floor == Tile_type.CHECK_POINTS:
        if DEBUG: print(f"Checkpoint detected at {maze.current_start_position_grid}!")
    elif tile.floor == Tile_type.SWAMP:
        if DEBUG: print(f"Swamp detected at {maze.current_start_position_grid}!")

    direction = maze.current_direction
    if direction == NORTH:
        tile.north_wall = maze.front_wall
        tile.west_wall = maze.left_wall
        tile.east_wall = maze.right_wall
    elif direction == WEST:
        tile.west_wall = maze.front_wall
        tile.south_wall = maze.left_wall
        tile.north_wall = maze.right_wall
    elif direction == SOUTH:
        tile.south_wall = maze.front_wall
        tile.east_wall = maze.left_wall
        tile.west_wall = maze.right_wall
    elif direction == EAST:
        tile.east_wall = maze.front_wall
        tile.north_wall = maze.left_wall
        tile.south_wall = maze.right_wall

    if (first_visit == 0) and not (maze.current_start_position_grid == (0, 0)):
        if direction == SOUTH: tile.north_wall = False
        if direction == NORTH: tile.south_wall = False
        if direction == EAST: tile.west_wall = False
        if direction == WEST: tile.east_wall = False

def get_front_tile_coords():
    x, z = maze.current_start_position_grid
    direction = determine_direction()
    
    if direction == NORTH:
        return (x, z - 1)
    elif direction == SOUTH:
        return (x, z + 1)
    elif direction == EAST:
        return (x + 1, z)
    elif direction == WEST:
        return (x - 1, z)
    return (x, z)

def create_tile(coordinates):
    if coordinates not in maze_map:
        maze_map[coordinates] = Maze_Tile()
        if DEBUG: print(f'Creating tile at {coordinates}')
    return maze_map[coordinates]

def explore():
    maze.left_wall = left_wall()
    maze.right_wall = right_wall()
    maze.front_wall = front_wall()
    maze.heading = get_heading()
    maze.current_start_position = gps_data()
    maze.current_direction = determine_direction()
    convert_grid()
    update_tile()
    return Robot_States.DECIDE

def decide():
    if not maze.left_wall and not maze.left_hole:
        if DEBUG: print("Decided to turn Left")
        return Robot_States.TURN_LEFT
    elif not maze.front_wall:
        if DEBUG: print("Decided to move forward")
        maze.left_hole = False
        return Robot_States.MOVE_FORWARD
    elif not maze.right_wall:
        if DEBUG: print("Decided to turn right")
        maze.left_hole = False
        return Robot_States.TURN_RIGHT
    else:
        if DEBUG: print("Decided to spin")
        maze.left_hole = False
        return Robot_States.SPIN

def turn_left():
    if not maze.state_initialised:
        cardinal_heading = snap_to_cardinal(maze.heading)
        maze.target_heading = wrap_angle(cardinal_heading + np.pi/2)
        maze.state_initialised = True

    how_far = yaw_difference(get_heading(), maze.target_heading)

    if np.abs(how_far) < YAW_TOLERENCE:
        stop()
        maze.state_initialised = False
        return Robot_States.MOVE_FORWARD
    else:
        rotate_left()
        return Robot_States.TURN_LEFT

def move_forward():
    if not maze.state_initialised:
        maze.state_initialised = True
        maze.current_start_position = gps_data()

    how_far = distance_travelled(maze.current_start_position, gps_data())

    r, g, b = get_floor_color_rgb()

    if r < 65 and g < 65 and b < 65:
        stop()
        coordinate = get_front_tile_coords()
        tile = create_tile(coordinate)
        tile.floor = Tile_type.BLACK_HOLE
        maze.state_initialised = False
        if DEBUG: print("Black Hole detected via RGB! Reversing...")
        return Robot_States.BACK_TO_CENTRE

    # 2. 检测沼泽 (Swamp): RGB 匹配棕色/黄褐色区域
    if 215 <= r <= 245 and 185 <= g <= 215 and 105 < b < 135:
        stop()
        coordinate = get_front_tile_coords()
        tile = create_tile(coordinate)
        tile.floor = Tile_type.SWAMP
        maze.state_initialised = False
        if DEBUG: print("Swamp detected via RGB! Reversing...")
        return Robot_States.BACK_TO_CENTRE

    # 3. 半墙 / 障碍物检测
    if how_far < (TILE_SIZE * 0.8) and min(f_ds_l.getValue(), f_ds_m.getValue(), f_ds_r.getValue()) < FRONT_WALL_TOLERANCE:
        stop()
        maze.state_initialised = False
        if DEBUG: print("Half wall detected! Reversing...")
        if maze.victim_found:
            delay(1500)
            victim_report()
        return Robot_States.BACK_TO_CENTRE

    # 4. 到达下一个格子或撞墙
    elif how_far > TILE_SIZE or min(f_ds_l.getValue(), f_ds_m.getValue(), f_ds_r.getValue()) < FRONT_WALL_TOLERANCE:
        stop()
        maze.state_initialised = False
        if maze.victim_found:
            delay(1500)
            victim_report()
        return Robot_States.EXPLORE
    else:
        drive_forward()
        return Robot_States.MOVE_FORWARD

def turn_right():
    if not maze.state_initialised:
        cardinal_heading = snap_to_cardinal(maze.heading)
        maze.target_heading = wrap_angle(cardinal_heading - np.pi/2)
        maze.state_initialised = True

    how_far = yaw_difference(get_heading(), maze.target_heading)

    if np.abs(how_far) < YAW_TOLERENCE:
        stop()
        maze.state_initialised = False
        if maze.left_hole:
            return Robot_States.EXPLORE
        else:
            return Robot_States.MOVE_FORWARD
    else:
        rotate_right()
        return Robot_States.TURN_RIGHT

def spin():
    if not maze.state_initialised:
        cardinal_heading = snap_to_cardinal(maze.heading)
        maze.target_heading = wrap_angle(cardinal_heading + np.pi)
        maze.state_initialised = True

    how_far = yaw_difference(get_heading(), maze.target_heading)

    if np.abs(how_far) < YAW_TOLERENCE:
        stop()
        maze.state_initialised = False
        return Robot_States.MOVE_FORWARD
    else:
        rotate_left()
        return Robot_States.SPIN

def back_to_centre():
    if not maze.state_initialised:
        maze.state_initialised = True
        maze.heading = get_heading()
        maze.left_hole = True

    how_far = distance_travelled(gps_data(), maze.current_start_position)  
    if np.abs(how_far) < TRAVEL_TOLERENCE:
        stop()
        maze.state_initialised = False
        return Robot_States.TURN_RIGHT
    else:
        drive_backward()
        return Robot_States.BACK_TO_CENTRE

def convert_grid():
    maze.current_start_position_grid = (
        round(((maze.current_start_position[0] - maze.start_position[0]) / TILE_SIZE)), 
        round((maze.current_start_position[2] - maze.start_position[2]) / TILE_SIZE)
    )

#-------------------END OF FUNCTIONS---------------
state = Robot_States.EXPLORE

robot.step(TIME_STEP)

maze.start_position = gps_data()

maze.start_position_grid = (0, 0)

maze.current_start_position_grid = (0, 0)

maze.current_direction = determine_direction()

tile = get_current_tile()

tile.floor = Tile_type.START

while robot.step(TIME_STEP) != -1:          # Main loop
    state = check_messages(state)
    if state == Robot_States.EXPLORE:
        state = explore()
        if DEBUG:
            dump_map()
    elif state == Robot_States.DECIDE:
        state = decide()
    elif state == Robot_States.TURN_LEFT:
        state = turn_left()
    elif state == Robot_States.MOVE_FORWARD:
        detect_victim()
        state = move_forward()
    elif state == Robot_States.TURN_RIGHT:
        state = turn_right()
    elif state == Robot_States.SPIN:
        state = spin()
    elif state == Robot_States.BACK_TO_CENTRE:
        state = back_to_centre()