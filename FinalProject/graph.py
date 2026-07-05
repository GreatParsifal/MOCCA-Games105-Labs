from typing import Dict, List, Optional

from bvh_loader import BVHMotion


class Edge:
    def __init__(self, label: str, dest: "Node"):
        self.label = label
        self.destination = dest


class Node:
    def __init__(self, id: int = -1, name: Optional[str] = None, motion: Optional[BVHMotion] = None):
        self.identity = id
        self.edges: List[Edge] = []
        self.name = name if name is not None else str(self.identity)
        self.motion = motion

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def add_edge(self, input_edge: Edge):
        self.edges.append(input_edge)

    def remove_edge(self, edge_id: int):
        self.edges.pop(edge_id)

    def get_edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]


class Graph:
    def __init__(self, graph_file: Optional[str] = None, animation_dir: str = "./motion_material/") -> None:
        self.nodes: List[Node] = []
        self.graph_file = graph_file
        self.motions: List[BVHMotion] = []
        self.animation_dir = animation_dir
        self.node_map: Dict[str, Node] = {}

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def add_node(self, node: Node):
        node.identity = self.n_nodes
        self.nodes.append(node)
        self.node_map[node.name] = node
        if node.motion is not None:
            self.motions.append(node.motion)
        return node.identity

    def get_node(self, name: str) -> Node:
        return self.node_map[name]

    def add_transition(self, src_name: str, dst_name: str, label: Optional[str] = None):
        src = self.get_node(src_name)
        dst = self.get_node(dst_name)
        edge_label = label if label is not None else f"{src_name}->{dst_name}"
        src.add_edge(Edge(edge_label, dst))

    def get_successor_names(self, name: str) -> List[str]:
        node = self.get_node(name)
        return [edge.destination.name for edge in node.edges]

    def load_motion_nodes(self, motion_names: List[str]):
        for motion_name in motion_names:
            motion = BVHMotion(self.animation_dir + motion_name)
            self.add_node(Node(name=motion_name, motion=motion))

    @classmethod
    def build_demo_motion_graph(cls) -> "Graph":
        graph = cls()
        motion_names = [
            "idle.bvh",
            "walk.bvh",
            "run.bvh",
            "jump.bvh",
            "turn_left.bvh",
            "turn_right.bvh",
            "excep_motion/dance.bvh",
            "excep_motion/kick.bvh",
            "excep_motion/backflip.bvh",
        ]
        graph.load_motion_nodes(motion_names)

        transitions = {
            "idle.bvh": ["idle.bvh", "walk.bvh", "run.bvh", "jump.bvh", "turn_left.bvh", "turn_right.bvh", "excep_motion/dance.bvh", "excep_motion/kick.bvh", "excep_motion/backflip.bvh"],
            "walk.bvh": ["idle.bvh", "walk.bvh", "run.bvh", "jump.bvh", "turn_left.bvh", "turn_right.bvh", "excep_motion/dance.bvh", "excep_motion/kick.bvh", "excep_motion/backflip.bvh"],
            "run.bvh": ["idle.bvh", "walk.bvh", "run.bvh", "jump.bvh", "turn_left.bvh", "turn_right.bvh", "excep_motion/dance.bvh", "excep_motion/kick.bvh", "excep_motion/backflip.bvh"],
            "jump.bvh": ["idle.bvh", "walk.bvh", "run.bvh"],
            "turn_left.bvh": ["idle.bvh", "walk.bvh", "run.bvh", "turn_left.bvh", "excep_motion/dance.bvh", "excep_motion/kick.bvh", "excep_motion/backflip.bvh"],
            "turn_right.bvh": ["idle.bvh", "walk.bvh", "run.bvh", "turn_right.bvh", "excep_motion/dance.bvh", "excep_motion/kick.bvh", "excep_motion/backflip.bvh"],
            "excep_motion/dance.bvh": ["idle.bvh", "walk.bvh", "run.bvh"],
            "excep_motion/kick.bvh": ["idle.bvh", "walk.bvh", "run.bvh"],
            "excep_motion/backflip.bvh": ["idle.bvh", "walk.bvh", "run.bvh"],
        }

        for src_name, dst_names in transitions.items():
            for dst_name in dst_names:
                graph.add_transition(src_name, dst_name)

        return graph
