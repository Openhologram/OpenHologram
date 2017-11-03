function varargout = fn_gui_Mesh(varargin)
%FN_GUI_MESH MATLAB code file for fn_gui_Mesh.fig
%      FN_GUI_MESH, by itself, creates a new FN_GUI_MESH or raises the existing
%      singleton*.
%
%      H = FN_GUI_MESH returns the handle to a new FN_GUI_MESH or the handle to
%      the existing singleton*.
%
%      FN_GUI_MESH('Property','Value',...) creates a new FN_GUI_MESH using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to fn_gui_Mesh_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      FN_GUI_MESH('CALLBACK') and FN_GUI_MESH('CALLBACK',hObject,...) call the
%      local function named CALLBACK in FN_GUI_MESH.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help fn_gui_Mesh

% Last Modified by GUIDE v2.5 03-Nov-2017 16:41:06

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @fn_gui_Mesh_OpeningFcn, ...
                   'gui_OutputFcn',  @fn_gui_Mesh_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% End initialization code - DO NOT EDIT
global filename;

% --- Executes just before fn_gui_Mesh is made visible.
function fn_gui_Mesh_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for fn_gui_Mesh
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes fn_gui_Mesh wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = fn_gui_Mesh_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in checkgpu.
function checkgpu_Callback(hObject, eventdata, handles)
% hObject    handle to checkgpu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkgpu
guidata(hObject, handles);



function shiftx_Callback(hObject, eventdata, handles)
% hObject    handle to shiftx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of shiftx as text
%        str2double(get(hObject,'String')) returns contents of shiftx as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function shiftx_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shiftx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function shifty_Callback(hObject, eventdata, handles)
% hObject    handle to shifty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of shifty as text
%        str2double(get(hObject,'String')) returns contents of shifty as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function shifty_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shifty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function shiftz_Callback(hObject, eventdata, handles)
% hObject    handle to shiftz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of shiftz as text
%        str2double(get(hObject,'String')) returns contents of shiftz as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function shiftz_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shiftz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function objectsize_Callback(hObject, eventdata, handles)
% hObject    handle to objectsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of objectsize as text
%        str2double(get(hObject,'String')) returns contents of objectsize as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function objectsize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to objectsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in loadbutton.
function loadbutton_Callback(hObject, eventdata, handles)
% hObject    handle to loadbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global filename;
[filename.loadName,filename.loadPath] = uigetfile('*.txt','Load file As');
filename.loadFile = strcat(filename.loadPath,filename.loadName);
set(handles.objectfile,'String',filename.loadFile);
guidata(hObject, handles);


function objectfile_Callback(hObject, eventdata, handles)
% hObject    handle to objectfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of objectfile as text
%        str2double(get(hObject,'String')) returns contents of objectfile as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function objectfile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to objectfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit34_Callback(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit34 as text
%        str2double(get(hObject,'String')) returns contents of edit34 as a double


% --- Executes during object creation, after setting all properties.
function edit34_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


function edit35_Callback(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit35 as text
%        str2double(get(hObject,'String')) returns contents of edit35 as a double


% --- Executes during object creation, after setting all properties.
function edit35_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in savecheck.
function savecheck_Callback(hObject, eventdata, handles)
% hObject    handle to savecheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of savecheck

if hObject.Value == 1
    set(handles.text57,'Visible','on');
    set(handles.holosavefile,'Visible','on');
    set(handles.holosavebutton,'Visible','on');
    set(handles.text58,'Visible','on');
    set(handles.reconsavefile,'Visible','on');
    set(handles.reconsavebutton,'Visible','on');
else if hObject.Value == 0
        set(handles.text57,'Visible','off');
        set(handles.holosavefile,'Visible','off');
        set(handles.holosavebutton,'Visible','off');
        set(handles.text58,'Visible','off');
        set(handles.reconsavefile,'Visible','off');
        set(handles.reconsavebutton,'Visible','off');
    end
end
guidata(hObject, handles);

% --- Executes on button press in figurecheck.
function figurecheck_Callback(hObject, eventdata, handles)
% hObject    handle to figurecheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of figurecheck
guidata(hObject, handles);


function lightx_Callback(hObject, eventdata, handles)
% hObject    handle to lightx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of lightx as text
%        str2double(get(hObject,'String')) returns contents of lightx as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function lightx_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lightx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function lighty_Callback(hObject, eventdata, handles)
% hObject    handle to lighty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of lighty as text
%        str2double(get(hObject,'String')) returns contents of lighty as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function lighty_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lighty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function lightz_Callback(hObject, eventdata, handles)
% hObject    handle to lightz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of lightz as text
%        str2double(get(hObject,'String')) returns contents of lightz as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function lightz_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lightz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function wavelength_Callback(hObject, eventdata, handles)
% hObject    handle to wavelength (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wavelength as text
%        str2double(get(hObject,'String')) returns contents of wavelength as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function wavelength_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wavelength (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Ny_Callback(hObject, eventdata, handles)
% hObject    handle to Ny (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Ny as text
%        str2double(get(hObject,'String')) returns contents of Ny as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Ny_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Ny (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Nx_Callback(hObject, eventdata, handles)
% hObject    handle to Nx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Nx as text
%        str2double(get(hObject,'String')) returns contents of Nx as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Nx_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Nx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function dy_Callback(hObject, eventdata, handles)
% hObject    handle to dy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of dy as text
%        str2double(get(hObject,'String')) returns contents of dy as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function dy_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function dx_Callback(hObject, eventdata, handles)
% hObject    handle to dx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of dx as text
%        str2double(get(hObject,'String')) returns contents of dx as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function dx_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in holosavebutton.
function holosavebutton_Callback(hObject, eventdata, handles)
% hObject    handle to holosavebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global filename;
[filename.holoName,filename.holoPath] = uiputfile('*.jpg','Save file As');
filename.holoFile = strcat(filename.holoPath,filename.holoName);
set(handles.holosavefile,'String',filename.holoFile);
guidata(hObject, handles);


function holosavefile_Callback(hObject, eventdata, handles)
% hObject    handle to holosavefile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of holosavefile as text
%        str2double(get(hObject,'String')) returns contents of holosavefile as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function holosavefile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to holosavefile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in reconsavebutton.
function reconsavebutton_Callback(hObject, eventdata, handles)
% hObject    handle to reconsavebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global filename
[filename.reconName,filename.reconPath] = uiputfile('*.jpg','Save file As');
filename.reconFile = strcat(filename.reconPath,filename.reconName);
set(handles.reconsavefile,'String',filename.reconFile);
guidata(hObject, handles);


function reconsavefile_Callback(hObject, eventdata, handles)
% hObject    handle to reconsavefile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of reconsavefile as text
%        str2double(get(hObject,'String')) returns contents of reconsavefile as a double
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function reconsavefile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to reconsavefile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Runbutton.
function Runbutton_Callback(hObject, eventdata, handles)
% hObject    handle to Runbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% obj=load(handles.);
% obj = fn_normalizeCenteringObj(obj);
% obj = fn_scaleShiftObj(obj, [objectSize, objectSize, objectSize], [shiftX, shiftY, shiftZ]);
global filename

fn_in_gui_Mesh(handles,filename);


% --- Executes on button press in continuousbutton.
function continuousbutton_Callback(hObject, eventdata, handles)
% hObject    handle to continuousbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of continuousbutton
guidata(hObject, handles);

% --- Executes on button press in flatbutton.
function flatbutton_Callback(hObject, eventdata, handles)
% hObject    handle to flatbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of flatbutton
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function shadingaxes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shadingaxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate shadingaxes


% --- Executes on mouse press over axes background.
function shadingaxes_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to shadingaxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function shadingtype_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shadingtype (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes when selected object is changed in shadingtype.
function shadingtype_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in shadingtype 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch get(hObject,'Tag')
    case 'continuousbutton'
        str = './teapot/con_recon.jpg';
    case 'flatbutton'
        str = './teapot/flat_recon.jpg';
end

imdata = imread(str);
axes(handles.shadingaxes);
imshow(imdata);


% --- Executes on key press with focus on savecheck and none of its controls.
function savecheck_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to savecheck (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function text57_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text57 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function shadingparampanel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shadingparampanel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1



function objecsize_Callback(hObject, eventdata, handles)
% hObject    handle to objectsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of objectsize as text
%        str2double(get(hObject,'String')) returns contents of objectsize as a double


% --- Executes during object creation, after setting all properties.
function objecsize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to objectsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
