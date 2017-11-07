function varargout = OpenHolo_transform(varargin)
% OPENHOLO_TRANSFORM MATLAB code for OpenHolo_transform.fig
%      OPENHOLO_TRANSFORM, by itself, creates a new OPENHOLO_TRANSFORM or raises the existing
%      singleton*.
%
%      H = OPENHOLO_TRANSFORM returns the handle to a new OPENHOLO_TRANSFORM or the handle to
%      the existing singleton*.
%
%      OPENHOLO_TRANSFORM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in OPENHOLO_TRANSFORM.M with the given input arguments.
%
%      OPENHOLO_TRANSFORM('Property','Value',...) creates a new OPENHOLO_TRANSFORM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before OpenHolo_transform_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to OpenHolo_transform_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help OpenHolo_transform

% Last Modified by GUIDE v2.5 30-Oct-2017 12:59:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @OpenHolo_transform_OpeningFcn, ...
                   'gui_OutputFcn',  @OpenHolo_transform_OutputFcn, ...
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


% --- Executes just before OpenHolo_transform is made visible.
function OpenHolo_transform_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to OpenHolo_transform (see VARARGIN)

% Choose default command line output for OpenHolo_transform
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes OpenHolo_transform wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = OpenHolo_transform_OutputFcn(hObject, eventdata, handles) 
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

z = str2double(get(handles.depth,'String'));                       % distance between object to hologram plane

x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
y = str2double(get(handles.resoly,'String'));                      % resolution of y axis

anglex = str2double(get(handles.anglex,'String'));                      % resolution of x axis
angley = str2double(get(handles.angley,'String'));                      % resolution of y axis

folder = get(handles.datapath,'String');
file = get(handles.filename,'String');
if get(handles.rbtnofah,'Value')
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    trans_H=cov_to_off(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],anglex,angley,l_x,l_y,wl);
elseif get(handles.rbtnhpo,'Value')
    red_rat = str2double(get(handles.redrat,'String'));                % reduction rate
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    trans_H=cov_to_hor(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],z,l_x,l_y,wl,red_rat);
elseif get(handles.rbtncac,'Value')
    wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
    wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
    wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
    f(1)=str2double(get(handles.f_r,'String'));
    f(2)=str2double(get(handles.f_g,'String'));
    f(3)=str2double(get(handles.f_b,'String'));
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    trans_H=chr_abb_comp(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],f,l_x,wl);
end

H=rec_holo(H,z,l_x,l_y,wl);
trans_H=rec_holo(trans_H,z,l_x,l_y,wl);

imagesc(handles.axes1,abs(H)), colormap gray;
imagesc(handles.axes2,abs(trans_H)), colormap gray;                      % threshold value

set(handles.input_state,'Value',1);
set(handles.trans_state,'Value',1);

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



function depth_Callback(hObject, eventdata, handles)
% hObject    handle to depth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of depth as text
%        str2double(get(hObject,'String')) returns contents of depth as a double


% --- Executes during object creation, after setting all properties.
function depth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to depth (see GCBO)
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

function redrat_Callback(hObject, eventdata, handles)
% hObject    handle to redrat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of redrat as text
%        str2double(get(hObject,'String')) returns contents of redrat as a double


% --- Executes during object creation, after setting all properties.
function redrat_CreateFcn(hObject, eventdata, handles)
% hObject    handle to redrat (see GCBO)
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


% --- Executes on selection change in input_state.
function input_state_Callback(hObject, eventdata, handles)
% hObject    handle to input_state (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns input_state contents as cell array
%        contents{get(hObject,'Value')} returns selected item from input_state
contents=cellstr(get(handles.input_state,'String'));
temp1=contents{get(handles.input_state,'Value')};

    % hologram parameter
    l_x = str2double(get(handles.sizex,'String'));                     % length of back ground (field of view)
    l_y = str2double(get(handles.sizey,'String'));                     % length of back ground (field of view)
    wl = str2double(get(handles.wavelength,'String'))*10^-9;           % wavelength

    z = str2double(get(handles.depth,'String'));                       % distance between object to hologram plane

    x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
    y = str2double(get(handles.resoly,'String'));                      % resolution of y axis
    
    folder = get(handles.datapath,'String');
    file = get(handles.filename,'String');
    
if strcmp(temp1,'Complex')
    if get(handles.rbtncac,'Value')
        wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
        wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
        wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
    end
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    H=rec_holo(H,z,l_x,l_y,wl);

    imagesc(handles.axes1,abs(H)), colormap gray;  
elseif strcmp(temp1,'Real')
    if get(handles.rbtncac,'Value')
        wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
        wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
        wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
    end
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    H=rec_holo(H,z,l_x,l_y,wl);

    imagesc(handles.axes1,real(H)), colormap gray;
elseif strcmp(temp1,'Imaginary')
    if get(handles.rbtncac,'Value')
        wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
        wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
        wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;         % wavelength
    end
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    H=rec_holo(H,z,l_x,l_y,wl);

    imagesc(handles.axes1,imag(H)), colormap gray;  
end


set(handles.zoom_input,'Value',0);
% --- Executes during object creation, after setting all properties.
function input_state_CreateFcn(hObject, eventdata, handles)
% hObject    handle to input_state (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in trans_state.
function trans_state_Callback(hObject, eventdata, handles)
% hObject    handle to trans_state (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns trans_state contents as cell array
%        contents{get(hObject,'Value')} returns selected item from trans_state
contents=cellstr(get(handles.trans_state,'String'));
temp1=contents{get(handles.trans_state,'Value')};

    % hologram parameter
    l_x = str2double(get(handles.sizex,'String'));                     % length of back ground (field of view)
    l_y = str2double(get(handles.sizey,'String'));                     % length of back ground (field of view)
    wl = str2double(get(handles.wavelength,'String'))*10^-9;           % wavelength

    z = str2double(get(handles.depth,'String'));                       % distance between object to hologram plane

    x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
    y = str2double(get(handles.resoly,'String'));                      % resolution of y axis
    
    anglex = str2double(get(handles.anglex,'String'));                      % resolution of x axis
    angley = str2double(get(handles.angley,'String'));                      % resolution of y axis
    folder = get(handles.datapath,'String');
    file = get(handles.filename,'String');
    
if strcmp(temp1,'Complex')
    if get(handles.rbtnofah,'Value')    
        trans_H=cov_to_off(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],anglex,angley,l_x,l_y,wl);
    elseif get(handles.rbtnhpo,'Value')
        red_rat = str2double(get(handles.redrat,'String'));                % reduction rate    
        trans_H=cov_to_hor(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],z,l_x,l_y,wl,red_rat);
    elseif get(handles.rbtncac,'Value')
        wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
        wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
        wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
        f(1)=str2double(get(handles.f_r,'String'));
        f(2)=str2double(get(handles.f_g,'String'));
        f(3)=str2double(get(handles.f_b,'String'));    
        trans_H=chr_abb_comp(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],f,l_x,wl);
    end

    trans_H=rec_holo(trans_H,z,l_x,l_y,wl);

    imagesc(handles.axes2,abs(trans_H)), colormap gray;  
elseif strcmp(temp1,'Real')
    if get(handles.rbtnofah,'Value')    
        trans_H=cov_to_off(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],anglex,angley,l_x,l_y,wl);
    elseif get(handles.rbtnhpo,'Value')
        red_rat = str2double(get(handles.redrat,'String'));                % reduction rate    
        trans_H=cov_to_hor(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],z,l_x,l_y,wl,red_rat);
    elseif get(handles.rbtncac,'Value')
        wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
        wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
        wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
        f(1)=str2double(get(handles.f_r,'String'));
        f(2)=str2double(get(handles.f_g,'String'));
        f(3)=str2double(get(handles.f_b,'String'));    
        trans_H=chr_abb_comp(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],f,l_x,wl);
    end

    trans_H=rec_holo(trans_H,z,l_x,l_y,wl);

    imagesc(handles.axes2,real(trans_H)), colormap gray;  
elseif strcmp(temp1,'Imaginary')    
    if get(handles.rbtnofah,'Value')    
        trans_H=cov_to_off(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],anglex,angley,l_x,l_y,wl);
    elseif get(handles.rbtnhpo,'Value')
        red_rat = str2double(get(handles.redrat,'String'));                % reduction rate    
        trans_H=cov_to_hor(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],z,l_x,l_y,wl,red_rat);
    elseif get(handles.rbtncac,'Value')
        wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
        wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
        wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
        f(1)=str2double(get(handles.f_r,'String'));
        f(2)=str2double(get(handles.f_g,'String'));
        f(3)=str2double(get(handles.f_b,'String'));    
        trans_H=chr_abb_comp(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],f,l_x,wl);
    end

    trans_H=rec_holo(trans_H,z,l_x,l_y,wl);
    imagesc(handles.axes2,imag(trans_H)), colormap gray;
end
set(handles.zoom_trans,'Value',0);

% --- Executes during object creation, after setting all properties.
function trans_state_CreateFcn(hObject, eventdata, handles)
% hObject    handle to trans_state (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function wl_r_Callback(hObject, eventdata, handles)
% hObject    handle to wl_r (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wl_r as text
%        str2double(get(hObject,'String')) returns contents of wl_r as a double


% --- Executes during object creation, after setting all properties.
function wl_r_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wl_r (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function f_r_Callback(hObject, eventdata, handles)
% hObject    handle to f_r (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f_r as text
%        str2double(get(hObject,'String')) returns contents of f_r as a double


% --- Executes during object creation, after setting all properties.
function f_r_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f_r (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function f_b_Callback(hObject, eventdata, handles)
% hObject    handle to f_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f_b as text
%        str2double(get(hObject,'String')) returns contents of f_b as a double


% --- Executes during object creation, after setting all properties.
function f_b_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function wl_g_Callback(hObject, eventdata, handles)
% hObject    handle to wl_g (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wl_g as text
%        str2double(get(hObject,'String')) returns contents of wl_g as a double


% --- Executes during object creation, after setting all properties.
function wl_g_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wl_g (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function f_g_Callback(hObject, eventdata, handles)
% hObject    handle to f_g (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f_g as text
%        str2double(get(hObject,'String')) returns contents of f_g as a double


% --- Executes during object creation, after setting all properties.
function f_g_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f_g (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function wl_b_Callback(hObject, eventdata, handles)
% hObject    handle to wl_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wl_b as text
%        str2double(get(hObject,'String')) returns contents of wl_b as a double


% --- Executes during object creation, after setting all properties.
function wl_b_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wl_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rbtnofah.
function rbtnofah_Callback(hObject, eventdata, handles)
% hObject    handle to rbtnofah (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbtnofah

if(get(handles.rbtnofah,'Value'))
    set(handles.rbtnhpo,'Value',0) 
    set(handles.rbtncac,'Value',0) 
end


% --- Executes on button press in rbtnhpo.
function rbtnhpo_Callback(hObject, eventdata, handles)
% hObject    handle to rbtnhpo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbtnhpo
if(get(handles.rbtnhpo,'Value'))
    set(handles.rbtnofah,'Value',0) 
    set(handles.rbtncac,'Value',0) 
end

% --- Executes on button press in rbtncac.
function rbtncac_Callback(hObject, eventdata, handles)
% hObject    handle to rbtncac (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbtncac
if(get(handles.rbtncac,'Value'))
    set(handles.rbtnofah,'Value',0) 
    set(handles.rbtnhpo,'Value',0) 
end


function anglex_Callback(hObject, eventdata, handles)
% hObject    handle to anglex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of anglex as text
%        str2double(get(hObject,'String')) returns contents of anglex as a double


% --- Executes during object creation, after setting all properties.
function anglex_CreateFcn(hObject, eventdata, handles)
% hObject    handle to anglex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function angley_Callback(hObject, eventdata, handles)
% hObject    handle to angley (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of angley as text
%        str2double(get(hObject,'String')) returns contents of angley as a double


% --- Executes during object creation, after setting all properties.
function angley_CreateFcn(hObject, eventdata, handles)
% hObject    handle to angley (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btnplot.
function btnplot_Callback(hObject, eventdata, handles)
% hObject    handle to btnplot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% hologram parameter
l_x = str2double(get(handles.sizex,'String'));                     % length of back ground (field of view)
l_y = str2double(get(handles.sizey,'String'));                     % length of back ground (field of view)
wl = str2double(get(handles.wavelength,'String'))*10^-9;           % wavelength

z = str2double(get(handles.depth,'String'));                       % distance between object to hologram plane

x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
y = str2double(get(handles.resoly,'String'));                      % resolution of y axis

anglex = str2double(get(handles.anglex,'String'));                      % resolution of x axis
angley = str2double(get(handles.angley,'String'));                      % resolution of y axis

folder = get(handles.datapath,'String');
file = get(handles.filename,'String');
if get(handles.rbtnofah,'Value')
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    trans_H=cov_to_off(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],anglex,angley,l_x,l_y,wl);
elseif get(handles.rbtnhpo,'Value')
    red_rat = str2double(get(handles.redrat,'String'));                % reduction rate
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    trans_H=cov_to_hor(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],z,l_x,l_y,wl,red_rat);
elseif get(handles.rbtncac,'Value')
    wl(1) = str2double(get(handles.wl_r,'String'))*10^-9;           % wavelength
    wl(2) = str2double(get(handles.wl_g,'String'))*10^-9;           % wavelength
    wl(3) = str2double(get(handles.wl_b,'String'))*10^-9;           % wavelength
    f(1)=str2double(get(handles.f_r,'String'));
    f(2)=str2double(get(handles.f_g,'String'));
    f(3)=str2double(get(handles.f_b,'String'));
    H=gen_holo_data(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp']);
    trans_H=chr_abb_comp(x,y,[folder,file,'_re.bmp'],[folder,file,'_im.bmp'],f,l_x,wl);
end

H=rec_holo(H,z,l_x,l_y,wl);
trans_H=rec_holo(trans_H,z,l_x,l_y,wl);

range = [1,x,0,1];

[~,~,color]=size(H);
for i=1:color
    figure(i),plot(1:x,abs(H(x/2+1,1:y,i)),'r',1:x,abs(trans_H(x/2+1,1:y,i)),'b'), axis(range), legend('original','transformed');
end


% --- Executes on slider movement.
function zoom_input_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

x = str2double(get(handles.resolx,'String'));                      % resolution of x axis
y = str2double(get(handles.resoly,'String'));                      % resolution of y axis
temp=get(handles.zoom_input,'Value');
set(handles.axes1,'ylim',[(1+((y-1)/2*temp)) (y-((y-1)/2*temp))]);
set(handles.axes1,'xlim',[(1+((x-1)/2*temp)) (x-((x-1)/2*temp))]);

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
