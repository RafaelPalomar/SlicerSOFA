import slicer
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from SofaSimulation import *

class OrganManipulationController(SimulationController):

    def __init__(self, parameterNode, parent=None):

        super(OrganManipulationController, self).__init__(parameterNode, parent)
        self._boxROI = None
        self._mouseInteractor = None

    def updateParameters(self) -> None:
        super(OrganManipulationController, self).updateParameters()
        if self.parameterNode:
            self._BoxROI.box = [self.parameterNode.getBoundaryROI()]
        if self.parameterNode.movingPointNode:
            self._mouseInteractor.position = [list(self.parameterNode.movingPointNode.GetNthControlPointPosition(0))*3]

    def updateScene(self) -> None:
        points_vtk = numpy_to_vtk(num_array=self._mechanicalObject.position.array(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(points_vtk)
        self.parameterNode.modelNode.GetUnstructuredGrid().SetPoints(vtk_points)

    def createScene(self, parameterNode) -> Sofa.Core.Node:
        from stlib3.scene import MainHeader, ContactHeader
        from stlib3.solver import DefaultSolver
        from stlib3.physics.deformable import ElasticMaterialObject
        from stlib3.physics.rigid import Floor
        from splib3.numerics import Vec3

        rootNode = Sofa.Core.Node()

        MainHeader(rootNode, plugins=[
            "Sofa.Component.IO.Mesh",
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
            "Sofa.Component.Topology.Container.Dynamic",
            "Sofa.Component.Engine.Select",
            "Sofa.Component.Constraint.Projective",
            "SofaIGTLink"
        ], dt=0.0, gravity=[0.0, 0.0, 0.0])

        rootNode.addObject('FreeMotionAnimationLoop', parallelODESolving=True, parallelCollisionDetectionAndFreeMotion=True)
        rootNode.addObject('GenericConstraintSolver', maxIterations=10, multithreading=True, tolerance=1.0e-3)

        femNode = rootNode.addChild('FEM')
        femNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
        femNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)

        self._container = femNode.addObject('TetrahedronSetTopologyContainer', name="Container")
        self._container.position = parameterNode.getModelPointsArray()
        self._container.tetrahedra = parameterNode.getModelCellsArray()

        femNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
        femNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
        femNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large")
        femNode.addObject('MeshMatrixMass', totalMass=1)

        fixedROI = femNode.addChild('FixedROI')
        self._BoxROI = fixedROI.addObject('BoxROI', template="Vec3", box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], drawBoxes=False,
                                          position="@../mstate.rest_position", name="FixedROI",
                                          computeTriangles=False, computeTetrahedra=False, computeEdges=False)
        fixedROI.addObject('FixedConstraint', indices="@FixedROI.indices")
        slicer.modules.boxROI = self._BoxROI

        collisionNode = femNode.addChild('Collision')
        collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
        collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
        collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")
        collisionNode.addObject('TriangleCollisionModel', name="collisionModel", proximity=0.001, contactStiffness=20)
        self._mechanicalObject = collisionNode.addObject('MechanicalObject', name='dofs', rest_position="@../mstate.rest_position")
        collisionNode.addObject('IdentityMapping', name='visualMapping')

        femNode.addObject('LinearSolverConstraintCorrection', linearSolver="@precond")

        attachPointNode = rootNode.addChild('AttachPoint')
        attachPointNode.addObject('PointSetTopologyContainer', name="Container")
        attachPointNode.addObject('PointSetTopologyModifier', name="Modifier")
        attachPointNode.addObject('MechanicalObject', name="mstate", template="Vec3d", drawMode=2, showObjectScale=0.01, showObject=False)
        self._mouseInteractor = attachPointNode.addObject('iGTLinkMouseInteractor', name="mouseInteractor", pickingType="constraint", reactionTime=20, destCollisionModel="@../FEM/Collision/collisionModel")

        return rootNode
