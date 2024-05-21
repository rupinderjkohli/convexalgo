# https://github.com/SiddhantSadangi/st_login_form/blob/main/src/st_login_form/__init__.py
# https://towardsdatascience.com/how-to-add-a-user-authentication-service-in-streamlit-a8b93bf02031
# https://github.com/mkhorasani/Streamlit-Authenticator/tree/main?tab=readme-ov-file#authenticatelogin

import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu

import yaml
from yaml.loader import SafeLoader

from algotrading_helper import *



# with open('./config.yaml') as file:
#         config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['pre-authorized']
# )
    
def on_change(key):
      selection = st.session_state[key]
      # st.write(f"Selection changed to {selection}")
      st.session_state['main_menu'] = selection
      st.session_state['selected_menu'] = selection
      # st.write("ON CHANGE DID I REACH HERE")
      return
  
def user_login_process():
    
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
    
    # ***USER LOGIN***
  
    if 'name' not in st.session_state:
        st.session_state['name'] = None
    if(('authentication_status' not in st.session_state)):
        st.session_state["authentication_status"] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'logout' not in st.session_state:
        st.session_state['logout'] = None
    if ('main_menu' not in st.session_state):
        st.session_state['main_menu'] = st.session_state.get('main_menu', 0)
    if ('selected_menu' not in st.session_state):  
        st.session_state['selected_menu'] = "Signals"
    if 'user_type' not in st.session_state:
        st.session_state['user_type'] = None
    if 'last_run' not in st.session_state:
        st.session_state['last_run'] = datetime.now() #since the app is being run the first time
  
    # st.write(st.session_state)
    
    name, authentication_status, username = authenticator.login()
     
    # st.write(name, authentication_status, username)
    # st.write("on login: ",st.session_state)
        
    if st.session_state["authentication_status"]:
        st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
        user_ticker = load_user_selected_options(username)
        authenticator.logout(location='sidebar')
        st.session_state['user_type'] = "RU"
        if (len(user_ticker) == 0): 
            show_menu("RU", "Setup Day")
    
            # st.write("Select to Setup Day option from the side menu for first time Ticker Setup")
        else: show_menu("RU", "Signals")
        return st.session_state['user_type']
            
        # # disable some menu options for new users
        # if username == 'guest':
        #     st.write(f'Welcome *{st.session_state["name"]}*')
        #     st.session_state['user_type'] = "GU"
        #     # call a function to disable some options menu
        #     show_menu("GU")
        #     if (len(user_ticker) == 0): 
        #         st.write("Select to Setup Day option from the side menu for first time Ticker Setup")
        #     return st.session_state['user_type']
        # else: 
        #     st.session_state['user_type'] = "RU"
        #     show_menu("RU")
        #     if (len(user_ticker) == 0): 
        #         st.write("Select to Setup Day option from the side menu for first time Ticker Setup")
        #     return st.session_state['user_type']
            
        # proceed further
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        st.session_state['user_type'] = None
        return st.session_state['user_type']
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
        st.session_state['user_type'] = None
        return st.session_state['user_type']
    
    # if (~st.session_state["authentication_status"]):
    #   try:
    #       if authenticator.reset_password(st.session_state["username"]):
    #           st.success('Password modified successfully')
    #   except Exception as e:
    #       st.error(e)
    
    # try:
    #   email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
    #   if email_of_registered_user:
    #       st.success('User registered successfully')
    # except Exception as e:
    #   st.error(e)
            
    # ***USER LOGIN DONE***

def show_menu(user, option):
    with st.sidebar:
        choose = option_menu("Convex Algos", ["Signals", "Status", "Trading Charts", "Change Logs", "---" ,"Algo Playground","---","Setup Day",],
                            icons=['camera fill', 'list-columns-reverse', 'bar-chart-line','person lines fill',"---" ,"battery-charging","---" ,'house', ],
                            menu_icon="app-indicator", 
                            default_index=0,
                            #  default_index=["Signals", "Status", "Trading Charts", "Change Logs", "---" ,"Setup Day",].index(st.session_state.selected_menu),
                            styles={
                            "container": {"padding": "5!important"}, #, "background-color": "#fafafa"},
                            "icon": {"color": "orange", "font-size": "25px"}, 
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#02ab21"},
                            
                        },
                            key='main_menu',
                            on_change=on_change
        )
        manual_select = option #"Signals"
        # st.write(st.session_state.get('main_menu', option))
        # Update session state based on selection
        st.session_state['selected_menu'] = choose
        
        # # Initialize session state
        # if 'main_menu' not in st.session_state:
        #     st.session_state.main_menu = 0 
        #     manual_select = st.session_state['main_menu']
        
        if st.session_state.get('main_menu', option):
            # st.session_state['main_menu'] = st.session_state.get('main_menu', 0)#+ 1) % 5
            # st.write("st.session_state.get('main_menu', option)",st.session_state.get('main_menu', option))
            manual_select = st.session_state['main_menu']
            # st.write(manual_select)
        else:
            manual_select = st.session_state.get('main_menu', 0) #None
    
    
    
# authenticator.login()