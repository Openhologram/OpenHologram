function varargout = OpenHolo_Reconstruction(varargin)
% OPENHOLO_RECONSTRUCTION MATLAB code for OpenHolo_Reconstruction.fig
%      OPENHOLO_RECONSTRUCTION, by itself, creates a new OPENHOLO_RECONSTRUCTION or raises the existing
%      singleton*.
%
%      H = OPENHOLO_RECONSTRUCTION returns the handle to a new OPENHOLO_RECONSTRUCTION or the handle to
%      the existing singleton*.
%
%      OPENHOLO_RECONSTRUCTION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in OPENHOLO_RECONSTRUCTION.M with the given input arguments.
%
%      OPENHOLO_RECONSTRUCTION('Property','Value',...) creates a new OPENHOLO_RECONSTRUCTION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before OpenHolo_Reconstruction_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to OpenHolo_Reconstruction_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help OpenHolo_Reconstruction

% Last Modified by GUIDE v2.5 30-Oct-2017 14:43:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @OpenHolo_Reconstruction_OpeningFcn, ...
                   'gui_OutputFcn',  @OpenHolo_Reconstruction_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
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


% --- Executes just before OpenHolo_Reconstruction is made visible.
function OpenHolo_Reconstruction_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to OpenHolo_Reconstruction (see VARARGIN)

% Choose default command line output for OpenHolo_Reconstruction
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes OpenHolo_Reconstruction wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = OpenHolo_Reconstruction_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function resolx_Callback(hObject, eventdata, handles)
% hObject    handle to resolx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of resolx as text
%        str2double(get(hObject,'String')) returns contents of resolx as a double


% --- Executes during object creation, after setting all properties.
function resolx_CreateFcn(hObject, eventdata, handles)
% hObject    handle to resolx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in btnTrans.
function btnTrans_Callback(hObject, eventdata, handles)
% hObject    handle to btnTrans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% hologram parameter
l_x = str2double(get(handles.sizex,'String'));                     % length of back ground (field of view)
l_y = str2double(get(handles.sizey,'String'));                     % length of back ground (field of view)
wl = str2double(get(handles.wavelength,'String'))*10^-9;           % wavelength

x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
y = str2double(get(handles.resoly,'String'));                      % resolution of y axis

folder = get(handles.datapath,'String');
file = get(handles.filename,'String');

zmin = str2double(get(handles.zmin,'String'));
zmax = str2double(get(handles.zmax,'String'));

samp = str2double(get(handles.samp,'String'));
th = str2double(get(handles.th,'String'));

if get(handles.rbtnsf,'Value')
    [H,z,F]=dep_shar(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],l_x,l_y,wl,zmax,zmin,samp,th);
    plot(handles.axes1,z,F), colormap gray;
    [~, index] = max(F);
    Z = z(index);
elseif get(handles.rbtnaxis,'Value')
    [H,Z]=dep_axi(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],l_x,l_y,wl);     
    plot(handles.axes1,Z), colormap gray;
    [~,Z] = max(Z);
    Z=-((Z-120)/10)/140+0.1;
end

imagesc(handles.axes2,abs(H)), colormap gray; 
set(handles.extracted_depth,'String',num2str(Z));
set(handles.reconstruction_state,'Value',1);

set(handles.zoom_input,'Value',0);
set(handles.zoom_trans,'Value',0);

function resoly_Callback(hObject, eventdata, handles)
% hObject    handle to resoly (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of resoly as text
%        str2double(get(hObject,'String')) returns contents of resoly as a double


% --- Executes during object creation, after setting all properties.
function resoly_CreateFcn(hObject, eventdata, handles)
% hObject    handle to resoly (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sizex_Callback(hObject, eventdata, handles)
% hObject    handle to sizex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sizex as text
%        str2double(get(hObject,'String')) returns contents of sizex as a double


% --- Executes during object creation, after setting all properties.
function sizex_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sizex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sizey_Callback(hObject, eventdata, handles)
% hObject    handle to sizey (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sizey as text
%        str2double(get(hObject,'String')) returns contents of sizey as a double


% --- Executes during object creation, after setting all properties.
function sizey_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sizey (see GCBO)
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

function datapath_Callback(hObject, eventdata, handles)
% hObject    handle to datapath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of datapath as text
%        str2double(get(hObject,'String')) returns contents of datapath as a double


% --- Executes during object creation, after setting all properties.
function datapath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to datapath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filename_Callback(hObject, eventdata, handles)
% hObject    handle to filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filename as text
%        str2double(get(hObject,'String')) returns contents of filename as a double


% --- Executes during object creation, after setting all properties.
function filename_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in reconstruction_state.
function reconstruction_state_Callback(hObject, eventdata, handles)
% hObject    handle to reconstruction_state (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns reconstruction_state contents as cell array
%        contents{get(hObject,'Value')} returns selected item from reconstruction_state
contents=cellstr(get(handles.reconstruction_state,'String'));
temp1=contents{get(handles.reconstruction_state,'Value')};

    % hologram parameter
    l_x = str2double(get(handles.sizex,'String'));                     % length of back ground (field of view)
    l_y = str2double(get(handles.sizey,'String'));                     % length of back ground (field of view)
    wl = str2double(get(handles.wavelength,'String'))*10^-9;           % wavelength

    z = str2double(get(handles.extracted_depth,'String'));                       % distance between object to hologram plane

    x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
    y = str2double(get(handles.resoly,'String'));                      % resolution of y axis
    
    folder = get(handles.datapath,'String');
    file = get(handles.filename,'String');   

if strcmp(temp1,'Complex')
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']); 
    H=rec_holo(H,z,l_x,l_y,wl);
    imagesc(handles.axes2,abs(H)), colormap gray;  
elseif strcmp(temp1,'Real')
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']); 
    H=rec_holo(H,z,l_x,l_y,wl);
    imagesc(handles.axes2,real(H)), colormap gray;   
elseif strcmp(temp1,'Imaginary')    
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']); 
    H=rec_holo(H,z,l_x,l_y,wl);
    imagesc(handles.axes2,imag(H)), colormap gray;   
end

% --- Executes during object creation, after setting all properties.
function reconstruction_state_CreateFcn(hObject, eventdata, handles)
% hObject    handle to reconstruction_state (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on slider movement.
function zoom_input_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

temp=get(handles.zoom_input,'Value');

x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
y = str2double(get(handles.resoly,'String'));                      % resolution of y axis

zmin = str2double(get(handles.zmin,'String'));
zmax = str2double(get(handles.zmax,'String'));

if get(handles.rbtnsf,'Value')
    set(handles.axes1,'xlim',[(zmin+(zmax*temp)) (zmax-(zmax*temp))]);
elseif get(handles.rbtnaxis,'Value')
    range = [1,x/4,0,y];
    set(handles.axes1,'xlim',[(1+(x/8*temp)) (x/4-(x/8*temp))]), axis(range);
end

% --- Executes during object creation, after setting all properties.
function zoom_input_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zoom_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on slider movement.
function zoom_trans_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_trans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
y = str2double(get(handles.resoly,'String'));                      % resolution of y axis

temp=get(handles.zoom_trans,'Value');
set(handles.axes2,'ylim',[(1+((y-1)/2*temp)) (y-((y-1)/2*temp))]);
set(handles.axes2,'xlim',[(1+((x-1)/2*temp)) (x-((x-1)/2*temp))]);

% --- Executes during object creation, after setting all properties.
function zoom_trans_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zoom_trans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function zmin_Callback(hObject, eventdata, handles)
% hObject    handle to zmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of zmin as text
%        str2double(get(hObject,'String')) returns contents of zmin as a double


% --- Executes during object creation, after setting all properties.
function zmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function zmax_Callback(hObject, eventdata, handles)
% hObject    handle to zmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of zmax as text
%        str2double(get(hObject,'String')) returns contents of zmax as a double


% --- Executes during object creation, after setting all properties.
function zmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function samp_Callback(hObject, eventdata, handles)
% hObject    handle to samp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of samp as text
%        str2double(get(hObject,'String')) returns contents of samp as a double


% --- Executes during object creation, after setting all properties.
function samp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to samp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function th_Callback(hObject, eventdata, handles)
% hObject    handle to th (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of th as text
%        str2double(get(hObject,'String')) returns contents of th as a double


% --- Executes during object creation, after setting all properties.
function th_CreateFcn(hObject, eventdata, handles)
% hObject    handle to th (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function extracted_depth_Callback(hObject, eventdata, handles)
% hObject    handle to extracted_depth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of extracted_depth as text
%        str2double(get(hObject,'String')) returns contents of extracted_depth as a double


% --- Executes during object creation, after setting all properties.
function extracted_depth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to extracted_depth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in rbtnsf.
function rbtnsf_Callback(hObject, eventdata, handles)
% hObject    handle to rbtnsf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbtnsf
if(get(handles.rbtnsf,'Value'))
    set(handles.rbtnaxis,'Value',0) 
end

% --- Executes on button press in rbtnaxis.
function rbtnaxis_Callback(hObject, eventdata, handles)
% hObject    handle to rbtnaxis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbtnaxis
if(get(handles.rbtnaxis,'Value'))
    set(handles.rbtnsf,'Value',0) 
end
