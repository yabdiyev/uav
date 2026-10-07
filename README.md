# Protocol Pro

Concept for an autonomous quadcopter on a [Holybro X500 V2](https://docs.holybro.com/drone-development-kit/px4-development-kit-x500v2) airframe. Cube/PX4 handles flight control; Jetson/ROS 2 is intended for perception and path planning.

Editable wiring diagram: [protocolpro.drawio](protocolpro.drawio).

## Hardware and connections

| Component | Connection |
|---|---|
| [CubePilot Cube Orange+](https://docs.px4.io/main/en/flight_controller/cubepilot_cube_orangeplus), standard carrier | Flight controller running PX4 |
| [Jetson Orin Nano Super Developer Kit](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/) | Cube TELEM2: UART / uXRCE-DDS |
| [CubePilot Here4](https://github.com/CubePilot/cubepilot-docs/blob/master/here-4/here-4-manual.md), RTK GNSS + compass | DroneCAN, chained after ARK Flow |
| [ARK Flow](https://docs.px4.io/main/en/dronecan/ark_flow), optical flow + ToF rangefinder | Cube CAN1: DroneCAN |
| 2× [RealSense D455](https://realsenseai.com/wp-content/uploads/dlm_uploads/2025/08/Intel-RealSense-D400-Series-Datasheet-August-2025.pdf) depth cameras | Jetson USB 3 |
| [RadioMaster TX16S 4-in-1](https://radiomasterrc.com/products/tx16s-mark-ii-radio-controller) | FrSky D16, 2.4 GHz wireless link to receiver |
| [FrSky R-XSR](https://www.frsky-rc.com/r-xsr/) | Cube RC IN: SBUS |
| [Holybro SiK Telemetry Radio V3](https://docs.holybro.com/radio/sik-telemetry-radio-v3), 915 MHz / 100 mW pair | Air unit: Cube TELEM1, UART / MAVLink. Ground unit: laptop USB / QGroundControl |
| 4× Holybro BLHeli S 20A ESCs | Cube AUX 1–4: DShot |
| 4× Holybro 2216 KV920 motors | One ESC per motor: three phase wires |
| CubePilot Power Brick Mini | Cube POWER1: 5 V supply + voltage/current sensing |
| Holybro X500 V2 PDB | Battery power to ESCs and Jetson DC input |
| 4S LiPo, 14.8 V nominal | Power Brick Mini input |

The X500 V2 kit includes the motors, ESCs and PDB listed above.

## Power

- Battery → Power Brick Mini → PDB → ESCs and Jetson DC input.
- Power Brick Mini → Cube POWER1, supplying regulated 5 V and battery measurements.
- Receiver, air radio and CAN sensors receive power through the Cube port cables; cameras receive USB power from the Jetson.

## Compatibility notes

- DShot on AUX 1–4 assumes DShot-capable ESC firmware. [PX4 guide](https://docs.px4.io/main/en/peripherals/dshot)
- Jetson UART uses 3.3 V logic, crossed TX/RX and common ground; leave Cube TELEM2 +5 V unconnected. uXRCE-DDS requires a compatible agent on the Jetson. [PX4 DDS guide](https://docs.px4.io/main/en/middleware/uxrce_dds)
- The CAN bus needs one 120 Ω termination at each end, including built-in resistors. ARK Flow termination stays off in the middle. [PX4 CAN guide](https://docs.px4.io/main/en/can/)
