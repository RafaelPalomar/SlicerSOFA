###################################################################################
# MIT License
#
# Copyright (c) 2024 Oslo University Hospital, Norway. All Rights Reserved.
# Copyright (c) 2024 NTNU, Norway. All Rights Reserved.
# Copyright (c) 2024 INRIA, France. All Rights Reserved.
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
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple

import inspect
from enum import Enum
from collections.abc import Iterable
from abc import abstractmethod
import numpy as np
import qt

import vtk
from vtk.util.numpy_support import numpy_to_vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import(
    vtkMRMLNode,
    vtkMRMLScalarVolumeNode,
    vtkMRMLSequenceBrowserNode
)

from SofaEnvironment import *

# -----------------------------------------------------------------------------
# Decorator: SofaParameterNodeWrapper
# -----------------------------------------------------------------------------
def SofaParameterNodeWrapper(cls):
    """
    Decorator function that wraps a class with additional SOFA-specific attributes and type checking,
    using `parameterNodeWrapper` and dataclasses.

    Args:
        cls (class): The class to wrap.

    Returns:
        class: The wrapped class with additional SOFA-specific functionality.
    """
    def __checkAndCreate__(cls, var, expectedType, defaultValue):
        """
        Internal function to check if a variable is of the expected type; if not, it sets a default value.

        Args:
            cls (class): The class being checked.
            var (str): The variable name.
            expectedType (type): The expected type of the variable.
            defaultValue: The default value to set if the variable is not present.

        Raises:
            TypeError: If the existing variable is not of the expected type.
        """
        if hasattr(cls, var):
            current_value = getattr(cls, var)
            if not isinstance(current_value, expectedType):
                raise TypeError(f"sofaParameterNodeWrapper: {var} should be of type {expectedType.__name__}, "
                                f"got {type(current_value).__name__}")
        else:
            # Sets attribute and creates a type annotation if not present
            setattr(cls, var, defaultValue)
            if not hasattr(cls, '__annotations__'):
                cls.__annotations__ = {}
            cls.__annotations__[var] = expectedType

    # Sets default simulation control parameters in the class
    __checkAndCreate__(cls, 'dt', float, 0.01)
    __checkAndCreate__(cls, 'totalSteps', int, -1)
    __checkAndCreate__(cls, 'currentStep', int, 0)
    __checkAndCreate__(cls, 'isSimulationRunning', bool, False)
    __checkAndCreate__(cls, 'sofaParameterNodeWrapped', bool, True)
    __checkAndCreate__(cls, 'simulationProgress', str, '')

    # Wrap the class using `parameterNodeWrapper` and `dataclass`
    wrapped_cls = dataclass(cls)
    wrapped_cls = parameterNodeWrapper(wrapped_cls)

    return wrapped_cls

# -----------------------------------------------------------------------------
# Class: SlicerSofa
# -----------------------------------------------------------------------------
class SlicerSofa(ScriptedLoadableModule):
    """
    Base class for the Slicer module using ScriptedLoadableModule, configuring metadata.
    """
    def __init__(self, parent):
        """
        Initialize the module with metadata.

        Args:
            parent (ScriptedLoadableModuleWidget): The parent object.
        """
        super().__init__(parent)
        self.parent.title = _("Slicer Sofa")  # Module title
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "")]  # Module category
        self.parent.dependencies = []  # Module dependencies
        self.parent.contributors = [
            "Rafael Palomar (Oslo University Hospital / NTNU, Norway)",
            "Paul Baksic (INRIA, France)",
            "Steve Pieper (Isomics, Inc., USA)",
            "Andras Lasso (Queen's University, Canada)",
            "Sam Horvath (Kitware, Inc., USA)",
            "Jean Christophe Fillion-Robin (Kitware, Inc., USA)"
        ]
        self.parent.helpText = _("""
        This is a support module to enable simulations using the SOFA framework
        See more information in <a href="https://github.com/RafaelPalomar/SlicerSOFA">module documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        Funded by Oslo University Hospital
        """)
        parent.hidden = True  # Hides this module from the Slicer module list

# -----------------------------------------------------------------------------
# Class: SlicerSofaWidget
# -----------------------------------------------------------------------------
class SlicerSofaWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.logic = None
        VTKObservationMixin.__init__(self)  # Needed for parameter node observation

    def setup(self) -> None:
        """
        Set up the module widget events when closing scenes and calls parent's setup
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Setup event connections for scene close events
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Handles the event when the scene starts to close.

        Args:
            caller: The caller object.
            event: The event triggered.
        """
        if self.logic:
            self.logic.stopSimulation()
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Handles the event when the scene has closed.

        Args:
            caller: The caller object.
            event: The event triggered.
        """
        if self.parent.isEntered:
            self.initializeParameterNode()


    def exit(self) -> None: 
        """
        Cleanup GUI connections when the module is exited.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def enter(self) -> None: 
        """
        Reconnect the GUI when the module is entered.
        """
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)


    def setParameterNode(self, parameterNode) -> None:
        """
        Sets the parameter node and connects GUI bindings.

        Args:
            parameterNode: The parameter node to set.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = parameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def updateWidgetOnSimulation(self, parentWidget=None):
        """
        Goes through all child widgets of the specified parent widget and checks
        for the 'SofaDisableOnSimulation' dynamic property.
        """
        slicer.modules.widget = parentWidget
        if parentWidget is None:
            parentWidget = self.parent  # Use the main widget if no parent is provided
        # Recursively checks and enables/disables widgets depending on the simulation status
        for child in parentWidget.findChildren(qt.QWidget):
            if child.property('SlicerDisableOnSimulation') is not None:
                disable = child.property('SlicerDisableOnSimulation')
                child.setEnabled(
                    (self._parameterNode.isSimulationRunning and not disable) or
                    (not self._parameterNode.isSimulationRunning and disable)
                )

# -----------------------------------------------------------------------------
# Class: SlicerSofaLogic
# -----------------------------------------------------------------------------
class SlicerSofaLogic(ScriptedLoadableModuleLogic):
    """
    Implements the computation logic for the Slicer SOFA module, handling the simulation lifecycle.
    """

    def __init__(self) -> None:
        """
        Initialize the logic with an optional scene creation function.

        """
        super().__init__()
        self.sofaMappings: List[Tuple[str, str, Callable, bool]] = []
        self.mrmlMappings: List[Tuple[str, str, Callable, bool]] = []
        self.recordSequenceFlags = {}
        self.runOnceFlags = {}
        self._sceneUp = False
        self._rootNode = None
        self._parameterNode = None
        self.ui = None

    @abstractmethod
    def createScene(self, parameterNode):
        """
        Abstract funtion for creating scene

        """
        pass

    def registerSOFAToMRMLMapping(self, fieldName: str, sofaPath: str, mappingFunction: Callable, runOnce: bool = False):
        """
        Register a mapping from MRML to SOFA.
        """
        self.sofaMappings.append((fieldName, sofaPath, mappingFunction, runOnce))
        if runOnce:
            self.runOnceFlags[mappingFunction] = False

    def registerMRMLToSOFAMapping(self, fieldName: str, sofaPath: str, mappingFunction: Callable, runOnce: bool = False):
        """
        Register a mapping from SOFA to MRML.
        """
        self.mrmlMappings.append((fieldName, sofaPath, mappingFunction, runOnce))
        if runOnce:
            self.runOnceFlags[mappingFunction] = False

    def setRecordSequenceFlag(self, fieldName: str, flagFunction: Callable):
        """
        Register a parameter-node field for sequence recording.

        The field is recorded only when flagFunction() is truthy at
        startSimulation() time.  Fields never passed to this method are never
        recorded -- registration is the opt-in, the flag is the switch.
        """
        self.recordSequenceFlags[fieldName] = flagFunction

    def _getSofaObjectByPath(self, path: str):
        """
        Helper method to retrieve a SOFA object by its path.
        """
        if not path:
            return self._rootNode
        parts = path.split('.')
        obj = self._rootNode
        for part in parts:
            if hasattr(obj, 'getChild') and obj.getChild(part):
                obj = obj.getChild(part)
            elif hasattr(obj, 'getObject') and obj.getObject(part):
                obj = obj.getObject(part)
            else:
                logging.warning(f"SOFA object '{part}' not found in path '{path}'.")
                return None
        return obj

    def setUi(self, ui):
        self.ui = ui

    def getUi(self):
        return self.ui

    def setupScene(self, parameterNode):
        """
        Initializes the SOFA simulation scene with parameter and root nodes.

        Args:
            parameterNode: The parameter node containing simulation parameters.
            rootNode (Sofa.Core.Node): The root node of the SOFA scene.

        Raises:
            ValueError: If rootNode is not a valid Sofa.Core.Node.
        """

        if parameterNode is None:
            raise ValueError("parameterNode can't be None")
        if not getattr(parameterNode, 'sofaParameterNodeWrapped', False):
            raise ValueError("parameterNode is not a valid parameterNode wrapped by the sofaParameterNodeWrapper")

        self._parameterNode = parameterNode
        self._rootNode = self.createScene(self._parameterNode)

        if not isinstance(self._rootNode, Sofa.Core.Node):
            raise ValueError("rootNode is not a valid Sofa.Core.Node root node")
        setattr(self._parameterNode, "_rootNode", self._rootNode)
        self.__updateSofa__()
        Sofa.Simulation.init(self._rootNode)
        self._sceneUp = True

    def startSimulation(self) -> None:
        """
        Starts the simulation by setting up the scene, resetting run-once flags,
        initializing sequence recording, and marking the simulation as running.
        """
        if self._parameterNode is None:
            raise ValueError("Parameter node has not been initialized.")
        self._saveState()
        self.resetRunOnceFlags()
        self.setupScene(self._parameterNode)
        self._parameterNode.currentStep = 0
        self._parameterNode.isSimulationRunning = True
        self.setupSequenceRecording()
        self._parameterNode.Modified()
        self.onSimulationStarted()

    def resetSimulation(self) -> None:
        self._restoreState()

    def stopSimulation(self) -> None:
        """
        Stops the simulation and halts sequence recording.
        """
        if self._sceneUp:
            Sofa.Simulation.unload(self._rootNode)
            self._parameterNode.isSimulationRunning = False
            self.stopSequenceRecording()
            self.onSimulationStopped()


    def onSimulationStarted(self):
        """
        Hook for module-specific logic when the simulation starts.
        Can be overridden in subclasses.
        """
        if self.ui is not None:
            self.ui.updateWidgetOnSimulation()

    def onSimulationStopped(self):
        """
        Hook for module-specific logic when the simulation stops.
        Can be overridden in subclasses.
        """
        if self.ui is not None:
            self.ui.updateWidgetOnSimulation()

    def updateSimulationProgress(self):
        if self._parameterNode.totalSteps < 0:
            self._parameterNode.simulationProgress = f"{self._parameterNode.currentStep}/\u221E"
        else:
            self._parameterNode.simulationProgress = f"{self._parameterNode.currentStep}/{self._parameterNode.totalSteps}"


    def simulationStep(self) -> None:
        """
        Executes a single simulation step, updating the simulation and MRML scenes.
        """
        if not self._sceneUp or not self._parameterNode.isSimulationRunning:
            return

        self.__updateSofa__()

        stepped = False
        if self._parameterNode.currentStep < self._parameterNode.totalSteps or self._parameterNode.totalSteps < 0:
            Sofa.Simulation.animate(self._rootNode, self._parameterNode.dt)
            self._parameterNode.currentStep += 1
            self.updateSimulationProgress()
            stepped = True
        else:
            self._parameterNode.isSimulationRunning = False
            self.onSimulationStopped()

        self.__updateMRML__()

        # One recorded frame per executed step: deterministic frame count,
        # and the final recorded frame equals the final state, so stopping
        # cannot rewind the proxies to anything but where they already are.
        if stepped and getattr(self, '_sequenceBrowserNode', None) is not None:
            self._sequenceBrowserNode.SaveProxyNodesState()

    def resetRunOnceFlags(self):
        """
        Resets all `runOnce` flags for sofaMappings and mrmlMappings.
        """
        for mappingFunction in self.runOnceFlags:
            self.runOnceFlags[mappingFunction] = False

    def initializeParameterNode(self):
        """
        Initializes the parameter node by retrieving it and resetting its parameters.
        """
        pass
        # if self._parameterNode is None:
        #     self._parameterNode = self.getParameterNode()
        # self.resetParameterNode()

    def resetParameterNode(self):
        """
        Resets simulation parameters in the parameter node to default values.
        """
        if self._parameterNode:
            self._parameterNode.currentStep = 0
            self._parameterNode.isSimulationRunning = False
            # Reset other parameters as needed
            self.onParameterNodeReset()

    def onParameterNodeReset(self):
        """
        Hook for module-specific parameter node reset logic.
        Can be overridden in subclasses.
        """
        pass

    def clean(self) -> None:
        """
        Cleans up the simulation resources by unloading the SOFA root node.
        """
        if self._sceneUp and self._rootNode:
            Sofa.Simulation.unload(self._rootNode)
            self._rootNode = None
            self._sceneUp = False
            self._parameterNode.isSimulationRunning = False

    @property
    def rootNode(self):
        """
        Property to access the SOFA root node.

        Returns:
            Sofa.Core.Node: The root node of the SOFA scene.
        """
        return self._rootNode

    def setupSequenceRecording(self):
        """
        Sets up sequence recording for the fields registered with
        setRecordSequenceFlag whose flag evaluates truthy.

        Recording is driven explicitly, one frame per executed simulation step
        (see simulationStep), NOT by the browser's own recording mode.  The
        browser's clock-based recording samples whenever a proxy is modified,
        throttled by wall time: a 200-step run recorded 26 frames on one
        machine, and SamplingAll over-recorded (601 frames for 200 steps,
        because the update pipeline modifies proxies several times per step).
        Either way the frame count depended on machine timing, which also made
        stopping hazardous: the browser snaps proxies back to the last recorded
        frame, so any read after stopSimulation() returned a state 0-14 steps
        stale.  With step-aligned capture the last recorded frame IS the final
        state, so that snap is the identity.
        """
        parameterNode = self.getParameterNode()
        if not parameterNode:
            raise ValueError("Parameter node is not initialized")

        # Only fields explicitly registered for recording, and only when their
        # flag says so.  Previously every mapped field was recorded
        # unconditionally and the registered flags were never read, so every
        # simulation paid the recording cost even with recordSequence off.
        recordableNodes = []
        for fieldName, flagFunction in self.recordSequenceFlags.items():
            try:
                enabled = bool(flagFunction())
            except Exception:  # noqa: BLE001 -- a broken flag must not kill the sim
                enabled = False
            if enabled and getattr(parameterNode, fieldName, None) is not None:
                recordableNodes.append(fieldName)

        if not recordableNodes:
            self._sequenceBrowserNode = None
            return  # No nodes to record

        self._sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', "SOFA Simulation Browser")
        for fieldName in recordableNodes:
            mrmlNode = getattr(parameterNode, fieldName, None)
            if isinstance(mrmlNode, vtkMRMLNode):
                sequenceNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', f"{mrmlNode.GetName()}_Sequence")
                self._sequenceBrowserNode.SetPlaybackActive(False)
                self._sequenceBrowserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
                self._sequenceBrowserNode.AddProxyNode(mrmlNode, sequenceNode, False)
                self._sequenceBrowserNode.SetRecording(sequenceNode, True)

        # Capture is driven from simulationStep; the browser's own clock-based
        # recording stays off.
        self._sequenceBrowserNode.SetRecordingActive(False)

    def stopSequenceRecording(self):
        """
        Stops sequence recording if a sequence browser node exists.

        With step-aligned capture the last recorded frame is the final
        simulation state, so the browser's snap-to-selected-item on stop is
        the identity -- reading a recorded proxy after stopSimulation() is
        safe.  (Under the previous clock-throttled recording it was not: the
        snap rewound proxies 0-14 steps depending on machine timing.)
        """
        if hasattr(self, '_sequenceBrowserNode') and self._sequenceBrowserNode is not None:
            self._sequenceBrowserNode.SetRecordingActive(False)
            logging.info("Sequence recording stopped.")


    def __updateSofa__(self):
        """
        Update SOFA nodes based on registered mappings.
        """
        pn = self.getParameterNode()
        for fieldName, sofaPath, mappingFunction, runOnce in self.mrmlMappings:
            if runOnce and self.runOnceFlags.get(mappingFunction, False):
                continue  # Skip if already run
            fieldValue = getattr(pn, fieldName, None)
            if fieldValue is None:
                continue  # Skip if parameter node field is None
            sofaObject = self._getSofaObjectByPath(sofaPath)
            if sofaObject is None:
                continue  # Skip if SOFA object not found
            mappingFunction(fieldValue, sofaObject)
            if runOnce:
                self.runOnceFlags[mappingFunction] = True

    def __updateMRML__(self):
        """
        Update MRML nodes based on registered mappings.
        """
        pn = self.getParameterNode()
        for fieldName, sofaPath, mappingFunction, runOnce in self.sofaMappings:
            if runOnce and self.runOnceFlags.get(mappingFunction, False):
                continue  # Skip if already run
            fieldValue = getattr(pn, fieldName, None)
            if fieldValue is None:
                continue  # Skip if parameter node field is None
            sofaObject = self._getSofaObjectByPath(sofaPath)
            if sofaObject is None:
                continue  # Skip if SOFA object not found
            mappingFunction(fieldValue, sofaObject)
            if runOnce:
                self.runOnceFlags[mappingFunction] = True

    @abstractmethod
    def setupMappings(self):
        """
        Abstract method for setting up mappings in derived classes.
        """
        pass


    def _saveState(self) -> None:
        """
        This function will be called on start simulation and it should be implemented
        on derived classes. The purpose of this function is to let the user save relevant
        objects for the reset of the simulation
        """
        pass

    def _restoreState(self) -> None:
        """
        This function will be called on simulation reset and it should be implemented on derive classes.
        The purpose of this function is to let the user restore a previously saved state of relevant
        objects during simulation reset.
        """
        pass
