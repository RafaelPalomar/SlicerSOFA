###################################################################################
# MIT License
#
# Copyright (c) 2024 Oslo University Hospital, Norway. All Rights Reserved.
# Copyright (c) 2024 NTNU, Norway. All Rights Reserved.
# Copyright (c) 2024 INRIA, France. All Rights Reserved.
# Copyright (c) 2024 Harvard Medical School. All Rights Reserved.
# Copyright (c) 2004 Brigham and Women's Hospital (BWH). All Rights Reserved.
# Copyright (c) 2024 Isomics, Inc., USA. All Rights Reserved.
# Copyright (c) 2024 Queen's University, Canada. All Rights Reserved.
# Copyright (c) 2024 Kitware Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################

import logging
import os
import qt
import vtk
import numpy as np

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.parameterNodeWrapper import parameterPack

import Sofa

from SlicerSofa import (
    SlicerSofaWidget,
    SlicerSofaLogic,
    SofaParameterNodeWrapper,
)

from SlicerSofaUtils.Mappings import (
    mrmlModelPolyToSofaTriangleTopologyContainer,
    mrmlMarkupsROIToSofaBoxROI,
    sofaMechanicalObjectToMRMLModelGrid,
    sofaMechanicalObjectToMRMLModelPoly,
    sofaSparseGridTopologyToMRMLModelGrid,
    arrayVectorFromMarkupsLinePoints,
    arrayFromModelGridCells,
    arrayFromMarkupsROIPoints,
)

#
# SparseGridSimulation
#

class SparseGridSimulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Sparse Grid Simulation"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital)",
            "Paul Baksic (INRIA)",
            "Steve Pieper (Isomics, Inc.)",
            "Andras Lasso (Queen's University)",
            "Sam Horvath (Kitware, Inc.)",
            "Jean-Christophe Fillion-Robin (Kitware, Inc.)",
            "Nazim Haouchine (Harvard Medical School / Brigham and Women's Hospital)"
        ]
        self.parent.helpText = """
        This is a Slicer-SOFA example module. The module allows creating a simulation based on sparse grid topology (which does not require tetrahedral meshes). In addition, the module allows the application of grid transforms derived from the simulation to medical images (e.g., segmentations and volumes).
        """
        self.parent.acknowledgementText = """
        This project was funded by Oslo University Hospital.
        """

        # Connect additional initialization after application startup
        slicer.app.connect("startupCompleted()", self.registerSampleData)

    def registerSampleData(self):
        """Register sample data sets in the Sample Data module."""

        import SampleData

        iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

        sofaDataURL= 'https://github.com/rafaelpalomar/SlicerSofaTestingData/releases/download/'

        # To ensure that the source code repository remains small (can be downloaded and installed quickly)
        # it is recommended to store data sets that are larger than a few MB in a Github release.

        # Right lung low poly tetrahedral mesh dataset
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            category='SOFA',
            sampleName='LiverSimulationScene',
            thumbnailFileName=os.path.join(iconsPath, 'LiverSimulationScene.png'),
            uris=sofaDataURL+ 'SHA256/19b38403d6ef301f2b049e3f32134962461b74dbccf31278938c2f7986371e89',
            fileNames='LiverSimulationScene.mrb',
            checksums='SHA256:19b38403d6ef301f2b049e3f32134962461b74dbccf31278938c2f7986371e89',
            nodeNames='LiverSimulationScene',
            loadFileType='SceneFile',
            loadFiles=True
        )

#
# CreateScene Function
#

def CreateScene(parameterNode) -> Sofa.Core.Node:
    """
    Creates the main SOFA scene with required components for simulation.
    """
    import Sofa.Core

    rootNode = Sofa.Core.Node("root")

    # # FIX: This is due to https://github.com/SofaDefrost/STLIB/issues/131
    # rootNode.addObject("RequiredPlugin", name="Sofa.Component.Visual")

    plugins=["Sofa.Component.IO.Mesh",
             "Sofa.Component.LinearSolver.Direct",
             "Sofa.Component.LinearSolver.Iterative",
             "Sofa.Component.Mapping.Linear",
             "Sofa.Component.Mass",
             "Sofa.Component.ODESolver.Backward",
             "Sofa.Component.Setting",
             "Sofa.Component.SolidMechanics.FEM.Elastic",
             "Sofa.Component.StateContainer",
             "Sofa.Component.Topology.Container.Dynamic",
             "Sofa.Component.Visual",
             "Sofa.GL.Component.Rendering3D",
             "Sofa.Component.AnimationLoop",
             "Sofa.Component.Collision.Detection.Algorithm",
             "Sofa.Component.Collision.Detection.Intersection",
             "Sofa.Component.Collision.Geometry",
             "Sofa.Component.Collision.Response.Contact",
             "Sofa.Component.Constraint.Lagrangian.Solver",
             "Sofa.Component.Constraint.Lagrangian.Correction",
             "Sofa.Component.LinearSystem",
             "Sofa.Component.MechanicalLoad",
             "MultiThreading",
             "Sofa.Component.SolidMechanics.Spring",
             "Sofa.Component.Constraint.Lagrangian.Model",
             "Sofa.Component.Mapping.NonLinear",
             "Sofa.Component.Topology.Container.Constant",
             "Sofa.Component.Topology.Mapping",
             "Sofa.Component.Engine.Select",
             "Sofa.Component.Constraint.Projective",
             "Sofa.Component.Topology.Container.Grid",
             ]

    for plugin_name in plugins:
        rootNode.addObject('RequiredPlugin', name=plugin_name)

    # Set gravity vector for the simulation (no gravity in this case)
    rootNode.gravity = [0, 0, 0]

    rootNode.addObject('DefaultAnimationLoop', parallelODESolving=True)
    rootNode.addObject('CollisionPipeline', depth=6, verbose=0, draw=0)
    rootNode.addObject('ParallelBruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('ParallelBVHNarrowPhase')
    rootNode.addObject('MinProximityIntersection', name="Proximity", alarmDistance=0.005, contactDistance=0.003)
    rootNode.addObject('CollisionResponse', name="Response", response="PenalityContactForceField")

    inputNode = rootNode.addChild('InputSurfaceNode')
    inputNode.addObject('TriangleSetTopologyContainer', name='Container', position=slicer.util.arrayFromModelPoints(parameterNode.modelNode), triangles=slicer.util.arrayFromModelPolyIds(parameterNode.modelNode).reshape(-1,4)[:,1:4])


    fem = rootNode.addChild('FEM')
    fem.addObject('SparseGridTopology', name='SparseGridTopology', n=parameterNode.sparseGridDimensions.array(), position="@../InputSurfaceNode/Container.position")
    fem.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    fem.addObject('CGLinearSolver', iterations=100, tolerance=1e-5, threshold=1e-5)
    fem.addObject('MechanicalObject', name='MO')
    fem.addObject('UniformMass', totalMass=0.5)
    fem.addObject('ParallelHexahedronFEMForceField', name="FEMForce", youngModulus=5, poissonRatio=0.40, method="large")

    surf = fem.addChild('Surf')
    surf.addObject('MeshTopology', position="@../../InputSurfaceNode/Container.position")
    surf.addObject('MechanicalObject', name='MechanicalObject', position="@../../InputSurfaceNode/Container.position")
    surf.addObject('TriangleCollisionModel', selfCollision=True)
    surf.addObject('LineCollisionModel')
    surf.addObject('PointCollisionModel')
    surf.addObject('BarycentricMapping')

    fem.addObject('BoxROI', name="FixedROI",
                  template="Vec3", box=arrayFromMarkupsROIPoints(parameterNode.boundaryROI), drawBoxes=False,
                  position="@MO.rest_position",
                  computeTriangles=False, computeTetrahedra=False, computeEdges=False)
    fem.addObject('FixedConstraint', indices="@FixedROI.indices")

    return rootNode

#
# ParameterPack Class: GridDimensions
#

@parameterPack
class GridDimensions:

    x: int
    y: int
    z: int

    def array(self) -> np.array:
        return np.array([self.x,self.y,self.z],dtype=np.int64)

#
# SparseGridSimulationParameterNode
#

@SofaParameterNodeWrapper
class SparseGridSimulationParameterNode:
    """
    Parameter class for the sparse grid simulation.
    Holds parameters but not mappings.
    """
    # MRML nodes
    modelNode: slicer.vtkMRMLModelNode
    boundaryROI: slicer.vtkMRMLMarkupsROINode
    gravityVector: slicer.vtkMRMLMarkupsLineNode
    sparseGridModelNode: slicer.vtkMRMLModelNode
    gridTransformNode: slicer.vtkMRMLGridTransformNode

    # Simulation parameters
    gravityMagnitude: float = 1.0  # Gravity strength
    recordSequence: bool = False   # Record sequence flag
    sparseGridDimensions: GridDimensions = GridDimensions(x=20, y=20, z=20)

#
# SparseGridSimulationWidget
#

class SparseGridSimulationWidget(SlicerSofaWidget):
    """
    UI widget for the Sparse Grid Simulation module.
    Manages user interactions and GUI elements.
    """

    def __init__(self, parent=None):
        SlicerSofaWidget.__init__(self, parent)
        self.logic = SparseGridSimulationLogic()
        self.timer = qt.QTimer()
        self.timer.timeout.connect(self.simulationStep)

    def setup(self):
        """
        Sets up the user interface, logic, and connections.
        """
        super().setup()

        # Load the UI from .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SparseGridSimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Set logic's UI reference
        self.logic.setUi(self)

        # connect UI buttons to logic functions
        self.ui.startSimulationPushButton.clicked.connect(self.startSimulation)
        self.ui.stopSimulationPushButton.clicked.connect(self.stopSimulation)
        self.ui.addBoundaryROIPushButton.clicked.connect(self.logic.addBoundaryROI)
        self.ui.addGravityVectorPushButton.clicked.connect(self.logic.addGravityVector)
        self.ui.addSparseGridModelNodePushButton.clicked.connect(self.logic.addSparseGridModelNode)
        self.ui.addGridTransformNodePushButton.clicked.connect(self.logic.addGridTransformNode)
        self.ui.resetSimulationPushButton.clicked.connect(self.logic.resetSimulation)

        # Initialize parameter node and GUI bindings
        self.setParameterNode(self.logic.getParameterNode())
        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)

    def cleanup(self):
        """
        Cleanup when the module widget is destroyed.
        Stops timers, simulation, and removes observers.
        """
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def initializeParameterNode(self):
        """
        Initializes and sets the parameter node in logic.
        """
        self.setParameterNode(self.logic.getParameterNode())
        self.logic.resetParameterNode()

    def updateSimulationGUI(self, caller, event):
        """
        Updates the GUI based on the simulation state.
        """
        parameterNode = self.logic.getParameterNode()
        isRunning = self.logic.isSimulationRunning()
        self.ui.startSimulationPushButton.setEnabled(not isRunning and parameterNode.modelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(isRunning)

    def startSimulation(self):
        """
        Starts the simulation and begins the timer for simulation steps.
        """
        self.logic.startSimulation()
        self.timer.start(0)

    def stopSimulation(self):
        """
        Stops the simulation and the timer.
        """
        self.timer.stop()
        self.logic.stopSimulation()

    def simulationStep(self):
        """
        Executes a single simulation step.
        """
        self.logic.simulationStep()

#
# SparseGridSimulationLogic
#

class SparseGridSimulationLogic(SlicerSofaLogic):
    """
    Logic class for the Sparse Grid Simulation.
    Handles scene setup, parameter node management, and simulation steps.
    """

    def __init__(self):
        super().__init__()
        self._rootNode = None
        self._parameterNode = None
        self._simulationRunning = False

    def createScene(self, parameterNode) -> Sofa.Core.Node:
        return CreateScene(parameterNode)

    def getParameterNode(self):
        """
        Retrieves or creates a wrapped parameter node.
        """
        if self._parameterNode is None:
            self._parameterNode = SparseGridSimulationParameterNode(super().getParameterNode())
        return self._parameterNode

    def resetParameterNode(self) -> None:
        """
        Resets simulation parameters in the parameter node to default values.
        """
        if self._parameterNode:
            self._parameterNode.modelNode = None
            self._parameterNode.boundaryROI = None
            self._parameterNode.gravityVector = None
            self._parameterNode.sparseGridModelNode = None
            self._parameterNode.gridTransformNode = None
            self._parameterNode.sparseGridDimensions = GridDimensions(x=10, y=10, z=10)
            self._parameterNode.gravityMagnitude = 1.0
            self._parameterNode.recordSequence = False
            self._parameterNode.dt = 0.01
            self._parameterNode.totalSteps = -1
            self._parameterNode.currentStep = 0

    def startSimulation(self):
        """
        Sets up the scene and starts the simulation.
        """
        self.setupMappings()
        self.setupScene(self.getParameterNode())
        super().startSimulation()
        self._simulationRunning = True
        self.getParameterNode().Modified()
        self._createGridTransformPipeline()

    def simulationStep(self):
        """
        Performs a simulation step and updates the MRML nodes.
        """
        super().simulationStep()
        self._updateProbingImage()

    def stopSimulation(self):
        """
        Stops the simulation.
        """
        super().stopSimulation()
        self._simulationRunning = False
        self.getParameterNode().Modified()

    def isSimulationRunning(self):
        """
        Returns whether the simulation is currently running.
        """
        return self._simulationRunning

    def addBoundaryROI(self):
        """
        Adds a boundary Region of Interest (ROI) based on the model's bounds.
        """
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        modelNode = self.getParameterNode().modelNode

        if modelNode and modelNode.GetPolyData():
            bounds = modelNode.GetPolyData().GetBounds()
            center = [(bounds[0] + bounds[1]) / 2.0,
                      (bounds[2] + bounds[3]) / 2.0,
                      (bounds[4] + bounds[5]) / 2.0]
            size = [abs(bounds[1] - bounds[0]) / 2.0,
                    abs(bounds[3] - bounds[2]) / 2.0,
                    abs(bounds[5] - bounds[4]) / 2.0]
            roiNode.SetCenter(center)
            roiNode.SetSize([s * 2 for s in size])

        self.getParameterNode().boundaryROI = roiNode

    def addSparseGridModelNode(self):
        """
        Adds a model node to hold the sparse grid from SOFA.
        """
        sparseGridModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', "Sparse Grid Model")
        sparseGridModelNode.CreateDefaultDisplayNodes()
        sparseGridModelNode.GetDisplayNode().SetVisibility(True)

        # Initialize the unstructured grid with displacement array
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        unstructuredGrid.SetPoints(points)
        displacementArray = vtk.vtkFloatArray()
        displacementArray.SetNumberOfComponents(3)
        displacementArray.SetName('Displacement')
        unstructuredGrid.GetPointData().AddArray(displacementArray)
        sparseGridModelNode.SetAndObserveMesh(unstructuredGrid)

        self.getParameterNode().sparseGridModelNode = sparseGridModelNode

    def addGravityVector(self):
        """
        Adds a gravity vector as a line in the scene.
        """
        gravityVector = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', "Gravity")
        gravityVector.CreateDefaultDisplayNodes()
        gravityVector.SetName("Gravity")
        gravityVector.GetMeasurement("length").SetEnabled(0)
        modelNode = self.getParameterNode().modelNode

        if modelNode and modelNode.GetPolyData():
            bounds = modelNode.GetPolyData().GetBounds()
            center = [(bounds[0] + bounds[1]) / 2.0,
                      (bounds[2] + bounds[3]) / 2.0,
                      (bounds[4] + bounds[5]) / 2.0]
            startPoint = [center[0], center[1], center[2]]
            endPoint = [center[0], center[1] - 10, center[2]]  # Example gravity vector

            gravityVector.AddControlPoint(startPoint)
            gravityVector.AddControlPoint(endPoint)

        self.getParameterNode().gravityVector = gravityVector

    def addGridTransformNode(self):
        """
        Adds a grid transform node to the scene.
        """
        gridTransformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLGridTransformNode', "Grid Transform")
        gridTransformNode.CreateDefaultDisplayNodes()
        self.getParameterNode().gridTransformNode = gridTransformNode

    # Mappings in Logic

    def setupMappings(self):
        """
        Registers mappings between MRML and SOFA nodes.
        """
        parameterNode = self.getParameterNode()

        if parameterNode is not None:
            # Register MRML-to-SOFA mappings
            self.registerMRMLToSOFAMapping('modelNode', 'InputSurfaceNode.Container', mrmlModelPolyToSofaTriangleTopologyContainer, runOnce=True)
            self.registerMRMLToSOFAMapping('boundaryROI', 'FEM.FixedROI', mrmlMarkupsROIToSofaBoxROI)
            self.registerMRMLToSOFAMapping('gravityVector', '', self.mrmlMarkupsLineToSofaGravityVector)
            self.registerMRMLToSOFAMapping('sparseGridDimensions', 'FEM.SparseGridTopology', self.gridDimensionsToSofaSparseGridTopology)

            # Register SOFA-to-MRML mappings
            self.registerSOFAToMRMLMapping('modelNode', 'FEM.Surf.MechanicalObject', sofaMechanicalObjectToMRMLModelPoly)
            self.registerSOFAToMRMLMapping('sparseGridModelNode', 'FEM.SparseGridTopology', sofaSparseGridTopologyToMRMLModelGrid, runOnce=True)
            self.registerSOFAToMRMLMapping('sparseGridModelNode', 'FEM.MO', self.sofaDisplacementToModelGridArray)
            self.registerSOFAToMRMLMapping('sparseGridModelNode', 'FEM.MO', sofaMechanicalObjectToMRMLModelGrid)

            # Set sequence recording flags
            self.setRecordSequenceFlag('modelNode', lambda: parameterNode.recordSequence)
            self.setRecordSequenceFlag('sparseGridModelNode', lambda: parameterNode.recordSequence)
            self.setRecordSequenceFlag('boundaryROI', lambda: parameterNode.recordSequence)
            self.setRecordSequenceFlag('gravityVector', lambda: parameterNode.recordSequence)
            self.setRecordSequenceFlag('gridTransformNode', lambda: parameterNode.recordSequence)

    def mrmlMarkupsLineToSofaGravityVector(self, gravityVectorNode, sofaRootNode):
        """
        Maps the gravity vector from MRML to the SOFA root node.
        """
        if gravityVectorNode is None:
            return
        gravityVector = arrayVectorFromMarkupsLinePoints(gravityVectorNode)
        magnitude = np.linalg.norm(np.array(gravityVector))
        normalizedGravityVector = gravityVector / magnitude if magnitude != 0 else gravityVector
        sofaRootNode.gravity = normalizedGravityVector * self.getParameterNode().gravityMagnitude

    def gridDimensionsToSofaSparseGridTopology(self, sparseGridDimensions, sofaNode):
        """
        Maps GridDimensions to SOFA's SparseGridTopology node.
        """
        if sparseGridDimensions:
            sofaNode.n = [sparseGridDimensions.x, sparseGridDimensions.y, sparseGridDimensions.z]

    def sofaDisplacementToModelGridArray(self, mrmlModelNode, sofaNode):
        """
        Updates the displacement array in the MRML model node from SOFA data.
        """
        mrmlModelNode.GetUnstructuredGrid().GetPointData().GetArray("Displacement").SetNumberOfTuples(int(sofaNode.position.size/3))
        displacementArray = slicer.util.arrayFromModelPointData(mrmlModelNode, "Displacement")
        displacementArray[:] = (sofaNode.position - sofaNode.rest_position)
        slicer.util.arrayFromModelPointsModified(mrmlModelNode)

    def _createGridTransformPipeline(self) -> None:

        self.probeGrid = vtk.vtkImageData()
        # NOTE: For now, the probe dimension is equal to the sparse grid dimension
        self.probeGrid.SetDimensions(self._parameterNode.sparseGridDimensions.x,
                                     self._parameterNode.sparseGridDimensions.y,
                                     self._parameterNode.sparseGridDimensions.z)
        self.probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        self.probeGrid.SetOrigin(0, 0, 0)
        self.probeGrid.SetSpacing(1, 1, 1)
        self.probeFilter = vtk.vtkProbeFilter()
        self.probeFilter.SetInputData(self.probeGrid)
        self.probeFilter.SetPassPointArrays(True)

        # The probe source is the sparse grid moved back to its rest positions,
        # rebuilt lazily on the first update; see _updateProbingImage.  Cleared
        # here because startSimulation creates a new SOFA scene, and with it a
        # sparse grid of possibly different size.
        self.restGrid = None

        # Provide TransformToParent directly instead of letting the rendering
        # pipeline invert TransformFromParent.  Inverting near steep
        # displacement gradients introduces transient singular regions that
        # cause the deformation to blow up at the grid boundaries.
        self.gridTransform = slicer.vtkOrientedGridTransform()
        self.displacementGrid = vtk.vtkImageData()
        # NOTE: The order is slices, rows, columns
        self.displacementGrid.SetDimensions(self._parameterNode.sparseGridDimensions.z,
                                            self._parameterNode.sparseGridDimensions.y,
                                            self._parameterNode.sparseGridDimensions.x)

        narray = np.zeros((self._parameterNode.sparseGridDimensions.x,
                           self._parameterNode.sparseGridDimensions.y,
                           self._parameterNode.sparseGridDimensions.z, 3))

        scalarType = vtk.util.numpy_support.get_vtk_array_type(narray.dtype)
        self.displacementGrid.AllocateScalars(scalarType, 3)
        self.gridTransform.SetDisplacementGridData(self.displacementGrid)
        self._parameterNode.gridTransformNode.SetAndObserveTransformToParent(self.gridTransform)

        displacementArray = vtk.util.numpy_support.vtk_to_numpy(
            self.displacementGrid.GetPointData().GetScalars()).reshape(narray.shape)
        displacementArray[:] = narray
        self.displacementGrid.Modified()

    def _updateProbingImage(self) -> None:
        """Resample the SOFA displacement field onto the regular grid transform.

        The field is installed as TransformToParent, so it must be a function of
        the *rest* coordinate: it answers "where did the undeformed point at x
        move to?".  Both the probed domain and the probe source therefore have
        to be the rest configuration.

        Neither is available directly.  sofaMechanicalObjectToMRMLModelPoly
        rewrites modelNode in place on every step, so modelNode.GetRASBounds()
        returns the *deformed* surface; sofaMechanicalObjectToMRMLModelGrid does
        the same to sparseGridModelNode's points.  Probing those makes the field
        a function of the deformed coordinate, so the sampling grid tracks the
        motion and cancels part of it, and the transform under-reports the
        deformation by an amount that grows with the deformation itself.  At
        small displacements the two domains nearly coincide and the error is
        second order, which is how this survived: a liver whose FEM surface moved
        49 mm produced a transform predicting 19 mm, correlated 0.15 with the
        mesh it came from.

        The rest positions are recoverable: the sparse grid carries both its
        deformed points and the Displacement that produced them.
        """
        sparseGrid = self._parameterNode.sparseGridModelNode.GetUnstructuredGrid()
        displacement = vtk.util.numpy_support.vtk_to_numpy(
            sparseGrid.GetPointData().GetArray("Displacement"))

        if self.restGrid is None:
            # Rest geometry is invariant, so build it once; only the
            # Displacement it carries changes from step to step.
            deformedPoints = vtk.util.numpy_support.vtk_to_numpy(
                sparseGrid.GetPoints().GetData())
            self.restGrid = vtk.vtkUnstructuredGrid()
            self.restGrid.DeepCopy(sparseGrid)
            self.restGrid.GetPoints().SetData(vtk.util.numpy_support.numpy_to_vtk(
                deformedPoints - displacement, deep=1))
            self.probeFilter.SetSourceData(self.restGrid)
        else:
            vtk.util.numpy_support.vtk_to_numpy(
                self.restGrid.GetPointData().GetArray("Displacement"))[:] = displacement
        self.restGrid.Modified()

        restBounds = self.restGrid.GetBounds()
        dimensions = self._parameterNode.sparseGridDimensions
        self.probeGrid.SetOrigin(restBounds[0], restBounds[2], restBounds[4])
        # n samples span the box in n-1 intervals.  Dividing by n instead leaves
        # the last cell of the model outside the field, where vtkGridTransform
        # can only clamp to the edge value.
        self.probeGrid.SetSpacing(
            (restBounds[1] - restBounds[0]) / max(dimensions.x - 1, 1),
            (restBounds[3] - restBounds[2]) / max(dimensions.y - 1, 1),
            (restBounds[5] - restBounds[4]) / max(dimensions.z - 1, 1))
        self.probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        self.probeGrid.Modified()
        self.probeFilter.Update()

        probeGrid = self.probeFilter.GetOutputDataObject(0)
        probeVTKArray = probeGrid.GetPointData().GetArray("Displacement")
        probeArray = vtk.util.numpy_support.vtk_to_numpy(probeVTKArray)
        probeArrayShape = (self._parameterNode.sparseGridDimensions.x,
                           self._parameterNode.sparseGridDimensions.y,
                           self._parameterNode.sparseGridDimensions.z,
                           3)
        probeArray = probeArray.reshape(probeArrayShape)
        gridArray = vtk.util.numpy_support.vtk_to_numpy(
            self.displacementGrid.GetPointData().GetScalars()).reshape(probeArrayShape)
        # Sign matches previous behavior: the old code stored -probeArray in
        # TransformFromParent and the rendering pipeline inverted it back to
        # +probeArray; storing directly into TransformToParent removes that
        # inversion, so the negation is no longer needed.
        gridArray[:] = probeArray
        self.displacementGrid.SetOrigin(probeGrid.GetOrigin())
        self.displacementGrid.SetSpacing(probeGrid.GetSpacing())
        self.displacementGrid.Modified()
        self._parameterNode.gridTransformNode.Modified()

#
# SparseGridSimulationTest
#

class SparseGridSimulationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for the SparseGridSimulation module.
    """

    def setUp(self):
        """Reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run the test case.

        NOTE: Slicer's test runner invokes runTest() only, so every test method
        must be called explicitly here.  A test_* method that is not listed is
        silently never executed.
        """
        self.setUp()
        self.test_SparseGridSimulation1()
        self.setUp()
        self.test_displacementGridIsStoredAsTransformToParent()
        self.setUp()
        self.test_displacementGridDimensionsMatchParameters()
        self.setUp()
        self.test_transformReproducesTheMeshDeformation()

    @staticmethod
    def _createInputSurfaceModel(radius=10.0, name="Test Surface"):
        """Build a triangulated sphere as simulation input.

        Generated procedurally rather than downloaded so the test is hermetic:
        no network access and no stored reference data.  CreateScene feeds the
        result to SOFA's TriangleSetTopologyContainer, which requires pure
        triangles, hence the vtkTriangleFilter.
        """
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(sphere.GetOutputPort())
        triangulate.Update()

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        modelNode.SetAndObservePolyData(triangulate.GetOutput())
        modelNode.CreateDefaultDisplayNodes()
        return modelNode

    def _buildLogic(self, dimensions=None):
        """Assemble a runnable SparseGridSimulation logic.

        NOTE: modelNode must be assigned *before* addBoundaryROI/addGravityVector,
        because both read parameterNode.modelNode to derive their geometry and
        silently fall back to degenerate defaults when it is None.
        """
        logic = SparseGridSimulationLogic()
        logic.resetParameterNode()
        parameterNode = logic.getParameterNode()

        parameterNode.modelNode = self._createInputSurfaceModel()
        if dimensions is not None:
            parameterNode.sparseGridDimensions = dimensions

        logic.addSparseGridModelNode()
        logic.addBoundaryROI()
        logic.addGravityVector()
        logic.addGridTransformNode()
        return logic

    def test_SparseGridSimulation1(self):
        """Basic test of the SparseGridSimulation module."""
        self.delayDisplay("Starting the test")

        logic = self._buildLogic()

        logic.startSimulation()
        for _ in range(10):
            logic.simulationStep()
            slicer.app.processEvents()
        logic.stopSimulation()

        self.delayDisplay("Test passed")

    def test_displacementGridIsStoredAsTransformToParent(self):
        """The displacement field must be installed as TransformToParent.

        Regression guard for the fix co-authored with @pieper: storing the field
        in TransformFromParent forces the rendering pipeline to invert it, and
        inverting near steep displacement gradients produces transient singular
        regions that blow up at the grid boundaries.  Slicer's documented
        convention also puts geometry-deforming transforms in ToParent.
        """
        self.delayDisplay("Starting TransformToParent test")

        logic = self._buildLogic()
        logic.startSimulation()
        logic.simulationStep()

        gridTransformNode = logic.getParameterNode().gridTransformNode
        toParent = gridTransformNode.GetTransformToParent()

        self.assertIsNotNone(toParent, "No TransformToParent installed on the grid transform node")
        self.assertIsInstance(toParent, slicer.vtkOrientedGridTransform)
        self.assertIsNotNone(toParent.GetDisplacementGrid(),
                             "TransformToParent carries no displacement grid")

        # The transform must be the one the logic pre-allocated, i.e. installed
        # directly rather than derived by inverting the opposite direction.
        self.assertIs(toParent, logic.gridTransform)

        logic.stopSimulation()
        self.delayDisplay("TransformToParent test passed")

    def test_transformReproducesTheMeshDeformation(self):
        """The grid transform must map each rest point to where the FEM moved it.

        The other tests in this file pin structure -- which object, which slot,
        what dimensions -- and no amount of that can tell a correct displacement
        field from one sampled over the wrong domain.  This asserts the meaning
        instead: evaluate TransformToParent at the undeformed surface vertices
        and require it to reproduce the deformation the simulation actually
        produced.  Everything downstream of this module (hardening onto a volume,
        resampling a segmentation) rides on the transform rather than on the
        mesh, so a transform that disagrees with its own mesh is silently wrong
        everywhere.

        Two details are load-bearing, because the defect being guarded against is
        second order in the displacement and a test that barely moves cannot see
        it:

        - addBoundaryROI covers the whole bounding box, which fixes nearly every
          node.  The ROI is narrowed to a basal slab so the sphere can bend.
        - gravityMagnitude is raised well above its 1.0 default.

        Regression guard for: the probe grid was positioned with
        modelNode.GetRASBounds() and sampled sparseGridModelNode's points, both
        of which the SOFA->MRML mappings rewrite in place to the *deformed*
        configuration.  The sampling grid then tracked the motion and cancelled
        part of it.  Measured on a liver before the fix: 49 mm of mean surface
        displacement produced a transform predicting 19 mm, correlated 0.15 with
        the mesh it was built from.
        """
        self.delayDisplay("Starting transform-reproduces-deformation test")

        logic = self._buildLogic()
        parameterNode = logic.getParameterNode()

        bounds = [0.0] * 6
        parameterNode.modelNode.GetPolyData().GetBounds(bounds)
        extent = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
        roiNode = parameterNode.boundaryROI
        roiNode.SetSize(2.0 * extent[0], 2.0 * extent[1], 0.4 * extent[2])
        roiNode.SetCenter((bounds[0] + bounds[1]) / 2.0,
                          (bounds[2] + bounds[3]) / 2.0,
                          bounds[4] + 0.1 * extent[2])
        parameterNode.gravityMagnitude = 500.0

        restPoints = slicer.util.arrayFromModelPoints(parameterNode.modelNode).copy()

        logic.startSimulation()
        for _ in range(50):
            logic.simulationStep()
            slicer.app.processEvents()

        deformedPoints = slicer.util.arrayFromModelPoints(parameterNode.modelNode).copy()
        actual = deformedPoints - restPoints
        actualMagnitude = np.linalg.norm(actual, axis=1)

        self.assertGreater(
            actualMagnitude.max(), 0.05 * max(extent),
            "the surface barely moved, so the test cannot distinguish a correct "
            "transform from a broken one -- raise gravityMagnitude or the step count")

        toParent = parameterNode.gridTransformNode.GetTransformToParent()
        predicted = np.array([toParent.TransformPoint(p.tolist()) for p in restPoints]) - restPoints
        error = np.linalg.norm(predicted - actual, axis=1)

        # The probe grid coincides with the FEM nodes and the surface rides the
        # hexahedra through a BarycentricMapping, so agreement should be near
        # exact; the tolerance only absorbs probe points outside the sparse grid.
        self.assertLess(
            error.mean(), 0.05 * actualMagnitude.mean(),
            "grid transform does not reproduce the mesh deformation: "
            "mean|error|=%.4f mm against mean displacement %.4f mm"
            % (error.mean(), actualMagnitude.mean()))

        # The field must be defined over the rest configuration it is indexed by.
        displacementGrid = toParent.GetDisplacementGrid()
        fieldBounds = displacementGrid.GetBounds()
        restBounds = [restPoints[:, 0].min(), restPoints[:, 0].max(),
                      restPoints[:, 1].min(), restPoints[:, 1].max(),
                      restPoints[:, 2].min(), restPoints[:, 2].max()]
        for axis in range(3):
            self.assertLessEqual(
                fieldBounds[2 * axis], restBounds[2 * axis] + 1e-6,
                "displacement field starts inside the rest model on axis %d" % axis)
            self.assertGreaterEqual(
                fieldBounds[2 * axis + 1], restBounds[2 * axis + 1] - 1e-6,
                "displacement field ends inside the rest model on axis %d" % axis)

        logic.stopSimulation()
        self.delayDisplay("Transform-reproduces-deformation test passed")

    def test_displacementGridDimensionsMatchParameters(self):
        """Displacement grid dimensions are the parameter dimensions reversed.

        VTK image point ordering runs fastest over the first dimension, so the
        module allocates (z, y, x) to match a numpy view shaped (x, y, z, 3).
        """
        self.delayDisplay("Starting grid dimensions test")

        logic = self._buildLogic()
        logic.startSimulation()
        logic.simulationStep()

        dimensions = logic.getParameterNode().sparseGridDimensions
        displacementGrid = logic.getParameterNode().gridTransformNode \
                                .GetTransformToParent().GetDisplacementGrid()

        self.assertEqual(displacementGrid.GetDimensions(),
                         (dimensions.z, dimensions.y, dimensions.x))
        self.assertEqual(displacementGrid.GetNumberOfScalarComponents(), 3)

        logic.stopSimulation()
        self.delayDisplay("Grid dimensions test passed")
