"""
SUMO Network Fixtures for Testing.

Provides mini SUMO networks and configurations for integration testing.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET


@pytest.fixture(scope="session")
def temp_sumo_dir(tmp_path_factory):
    """Create temporary directory for SUMO network files."""
    return tmp_path_factory.mktemp("sumo_networks")


@pytest.fixture
def mini_network_2x2_config(temp_sumo_dir) -> Dict[str, Path]:
    """Create a minimal 2x2 intersection network for testing."""
    base_dir = temp_sumo_dir / "mini_2x2"
    base_dir.mkdir(exist_ok=True)
    
    # Create nodes file
    nodes_file = base_dir / "nodes.nod.xml"
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="n0" x="0.0" y="0.0" type="priority"/>
    <node id="n1" x="100.0" y="0.0" type="priority"/>
    <node id="n2" x="0.0" y="100.0" type="priority"/>
    <node id="n3" x="100.0" y="100.0" type="priority"/>
</nodes>"""
    nodes_file.write_text(nodes_xml)
    
    # Create edges file
    edges_file = base_dir / "edges.edg.xml"
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="e0" from="n0" to="n1" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e1" from="n1" to="n0" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e2" from="n2" to="n3" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e3" from="n3" to="n2" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e4" from="n0" to="n2" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e5" from="n2" to="n0" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e6" from="n1" to="n3" priority="1" numLanes="1" speed="13.89"/>
    <edge id="e7" from="n3" to="n1" priority="1" numLanes="1" speed="13.89"/>
</edges>"""
    edges_file.write_text(edges_xml)
    
    # Create network file
    net_file = base_dir / "network.net.xml"
    # In real scenario, would run netconvert, but for testing we'll mock
    
    # Create sumocfg file
    cfg_file = base_dir / "test.sumocfg"
    cfg_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""
    cfg_file.write_text(cfg_xml)
    
    return {
        "config_file": cfg_file,
        "nodes_file": nodes_file,
        "edges_file": edges_file,
        "net_file": net_file,
        "base_dir": base_dir
    }


@pytest.fixture
def single_intersection_config(temp_sumo_dir) -> Dict[str, Path]:
    """Create a single intersection configuration."""
    base_dir = temp_sumo_dir / "single_intersection"
    base_dir.mkdir(exist_ok=True)
    
    # Simplified single intersection
    cfg_file = base_dir / "single.sumocfg"
    cfg_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""
    cfg_file.write_text(cfg_xml)
    
    return {
        "config_file": cfg_file,
        "base_dir": base_dir
    }


@pytest.fixture
def mock_sumo_connection():
    """Mock SUMO TraCI connection for testing."""
    from unittest.mock import Mock, MagicMock
    
    mock_traci = MagicMock()
    
    # Mock traffic light methods
    mock_traci.trafficlight.getIDList.return_value = ['tl0', 'tl1']
    mock_traci.trafficlight.getControlledLinks.return_value = [
        [('e0_0', 'e1_0', 0), ('e2_0', 'e3_0', 0)],
        [('e4_0', 'e5_0', 0), ('e6_0', 'e7_0', 0)]
    ]
    mock_traci.trafficlight.getRedYellowGreenState.return_value = 'GGGG'
    mock_traci.trafficlight.setRedYellowGreenState = Mock()
    
    # Mock vehicle methods
    mock_traci.vehicle.getIDList.return_value = ['veh0', 'veh1']
    mock_traci.vehicle.getWaitingTime.return_value = 5.0
    mock_traci.vehicle.getLaneID.return_value = 'e0_0'
    
    # Mock edge methods
    mock_traci.edge.getWaitingTime.return_value = 10.0
    mock_traci.edge.getLastStepVehicleNumber.return_value = 5
    
    # Mock simulation methods
    mock_traci.simulation.getTime.return_value = 0.0
    mock_traci.simulation.step = Mock()
    mock_traci.simulation.getMinExpectedNumber.return_value = 0
    
    mock_traci.start = Mock()
    mock_traci.close = Mock()
    
    return mock_traci


@pytest.fixture
def sumo_route_generator():
    """Generate SUMO route files for testing."""
    def _generate_routes(
        base_dir: Path,
        num_vehicles: int = 100,
        vehicle_rate: float = 0.1,
        duration: int = 3600
    ) -> Path:
        """Generate a route file."""
        routes_file = base_dir / "routes.rou.xml"
        
        routes_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'
        
        # Generate vehicles
        time = 0.0
        veh_id = 0
        while time < duration and veh_id < num_vehicles:
            routes_xml += f'    <vehicle id="veh{veh_id}" depart="{time:.1f}">\n'
            routes_xml += '        <route edges="e0 e1"/>\n'
            routes_xml += '    </vehicle>\n'
            time += 1.0 / vehicle_rate
            veh_id += 1
        
        routes_xml += '</routes>'
        routes_file.write_text(routes_xml)
        
        return routes_file
    
    return _generate_routes

