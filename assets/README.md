# Assets

The dual-Panda MuJoCo environment resolves the local PyBullet `franka_panda/panda.urdf`
and its mesh directory at runtime, so the repository does not duplicate those
licensed mesh files. The two-peg workpiece and two-hole receiver are generated
procedurally from `configs/task.yaml`, preserving the existing peg/hole dimensions.
Put future robot URDF/USD files under this directory and keep their meshes and
textures beside the source asset.
