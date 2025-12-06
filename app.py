import io
import streamlit as st
from PIL import Image
from yolo_infer import YoloDiseaseDetector



# Page configuration with modern settings
st.set_page_config(
    page_title="Calamansi Disease Detector", 
    page_icon="🍃", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with Dark/Light mode support
st.markdown("""
	<style>

    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for theming */
    :root {
        --bg-primary: #f0f8f0;
        --bg-secondary: #e8f5e8;
        --bg-card: #ffffff;
        --text-primary: #1b5e20;
        --text-secondary: #2e7d32;
        --text-muted: #4caf50;
        --accent-primary: #2e7d32;
        --accent-secondary: #4caf50;
        --accent-light: #8bc34a;
        --shadow-color: rgba(46, 125, 50, 0.1);
        --border-color: #c8e6c9;
    }
    
    /* Dark mode variables - Auto detect browser preference */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #000000;
            --bg-secondary: #1a1a1a;
            --bg-card: #2d2d2d;
            --text-primary: #e8f5e8;
            --text-secondary: #c8e6c9;
            --text-muted: #a5d6a7;
            --accent-primary: #66bb6a;
            --accent-secondary: #81c784;
            --accent-light: #a5d6a7;
            --shadow-color: rgba(102, 187, 106, 0.2);
            --border-color: #4caf50;
        }
        
        /* Dark mode leaf pattern - detailed realistic leaves */
        [data-testid="stAppViewContainer"] {
            background-image: 
                url("data:image/svg+xml,%3Csvg width='300' height='300' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='leaf-pattern-dark' x='0' y='0' width='300' height='300' patternUnits='userSpaceOnUse'%3E%3Cg opacity='0.1'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.08' transform='translate(140,70) rotate(45)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.06' transform='translate(220,180) rotate(-30)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.05' transform='translate(40,50) rotate(75)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.04' transform='translate(250,50) rotate(120)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23leaf-pattern-dark)'/%3E%3C/svg%3E");
        }
        
        .main .block-container {
            background-image: 
                url("data:image/svg+xml,%3Csvg width='400' height='400' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='leaf-pattern-dark-light' x='0' y='0' width='400' height='400' patternUnits='userSpaceOnUse'%3E%3Cg opacity='0.06'%3E%3Cpath d='M90,160 C70,125 50,115 38,125 C26,135 32,155 52,180 C72,205 95,190 105,165 C115,140 110,130 90,160 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='74' y1='150' x2='38' y2='125' stroke='%2381c784' stroke-width='0.35'/%3E%3Cline x1='74' y1='150' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='70' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='80' y2='145' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='85' y2='155' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='92' y2='165' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='65' y1='135' x2='58' y2='128' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='65' y1='135' x2='62' y2='131' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='75' y2='138' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='78' y2='140' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='85' y1='155' x2='82' y2='150' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.05' transform='translate(240,90) rotate(50)'%3E%3Cpath d='M90,160 C70,125 50,115 38,125 C26,135 32,155 52,180 C72,205 95,190 105,165 C115,140 110,130 90,160 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='74' y1='150' x2='38' y2='125' stroke='%2381c784' stroke-width='0.35'/%3E%3Cline x1='74' y1='150' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='70' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='80' y2='145' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='85' y2='155' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='92' y2='165' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='65' y1='135' x2='58' y2='128' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='65' y1='135' x2='62' y2='131' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='75' y2='138' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='78' y2='140' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='85' y1='155' x2='82' y2='150' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.04' transform='translate(300,220) rotate(-40)'%3E%3Cpath d='M90,160 C70,125 50,115 38,125 C26,135 32,155 52,180 C72,205 95,190 105,165 C115,140 110,130 90,160 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='74' y1='150' x2='38' y2='125' stroke='%2381c784' stroke-width='0.35'/%3E%3Cline x1='74' y1='150' x2='65' y2='135' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='70' y2='142' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='80' y2='145' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='85' y2='155' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='92' y2='165' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='65' y1='135' x2='58' y2='128' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='65' y1='135' x2='62' y2='131' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='75' y2='138' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='78' y2='140' stroke='%2381c784' stroke-width='0.2'/%3E%3Cline x1='85' y1='155' x2='82' y2='150' stroke='%2381c784' stroke-width='0.2'/%3E%3C/g%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23leaf-pattern-dark-light)'/%3E%3C/svg%3E");
        }
    }
    
    /* Manual dark mode override */
    [data-theme="dark"] {
        --bg-primary: #000000;
        --bg-secondary: #1a1a1a;
        --bg-card: #2d2d2d;
        --text-primary: #e8f5e8;
        --text-secondary: #c8e6c9;
        --text-muted: #a5d6a7;
        --accent-primary: #66bb6a;
        --accent-secondary: #81c784;
        --accent-light: #a5d6a7;
        --shadow-color: rgba(102, 187, 106, 0.2);
        --border-color: #4caf50;
    }
    
    /* Dark theme leaf pattern - realistic leaves */
    [data-theme="dark"] [data-testid="stAppViewContainer"] {
        background-image: 
            url("data:image/svg+xml,%3Csvg width='250' height='250' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='leaf-pattern-dark-theme' x='0' y='0' width='250' height='250' patternUnits='userSpaceOnUse'%3E%3Cg opacity='0.1'%3E%3Cpath d='M60,120 C50,100 40,95 35,100 C30,105 35,115 45,125 C55,135 65,130 70,120 C75,110 70,105 60,120 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='52' y1='112' x2='45' y2='100' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='50' y2='105' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='55' y2='108' stroke='%2381c784' stroke-width='0.3'/%3E%3C/g%3E%3Cg opacity='0.08' transform='translate(120,60) rotate(45)'%3E%3Cpath d='M60,120 C50,100 40,95 35,100 C30,105 35,115 45,125 C55,135 65,130 70,120 C75,110 70,105 60,120 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='52' y1='112' x2='45' y2='100' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='50' y2='105' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='55' y2='108' stroke='%2381c784' stroke-width='0.3'/%3E%3C/g%3E%3Cg opacity='0.06' transform='translate(180,150) rotate(-30)'%3E%3Cpath d='M60,120 C50,100 40,95 35,100 C30,105 35,115 45,125 C55,135 65,130 70,120 C75,110 70,105 60,120 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='52' y1='112' x2='45' y2='100' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='50' y2='105' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='55' y2='108' stroke='%2381c784' stroke-width='0.3'/%3E%3C/g%3E%3Cg opacity='0.05' transform='translate(30,40) rotate(75)'%3E%3Cpath d='M60,120 C50,100 40,95 35,100 C30,105 35,115 45,125 C55,135 65,130 70,120 C75,110 70,105 60,120 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.5'/%3E%3Cline x1='52' y1='112' x2='45' y2='100' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='50' y2='105' stroke='%2381c784' stroke-width='0.3'/%3E%3Cline x1='52' y1='112' x2='55' y2='108' stroke='%2381c784' stroke-width='0.3'/%3E%3C/g%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23leaf-pattern-dark-theme)'/%3E%3C/svg%3E");
    }
    
    [data-theme="dark"] .main .block-container {
        background-image: 
            url("data:image/svg+xml,%3Csvg width='350' height='350' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='leaf-pattern-dark-theme-light' x='0' y='0' width='350' height='350' patternUnits='userSpaceOnUse'%3E%3Cg opacity='0.06'%3E%3Cpath d='M80,140 C65,115 50,110 42,118 C34,126 40,140 55,155 C70,170 85,160 92,145 C99,130 95,120 80,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='68' y1='130' x2='55' y2='115' stroke='%2381c784' stroke-width='0.25'/%3E%3Cline x1='68' y1='130' x2='65' y2='120' stroke='%2381c784' stroke-width='0.25'/%3E%3Cline x1='68' y1='130' x2='72' y2='125' stroke='%2381c784' stroke-width='0.25'/%3E%3C/g%3E%3Cg opacity='0.05' transform='translate(200,80) rotate(50)'%3E%3Cpath d='M80,140 C65,115 50,110 42,118 C34,126 40,140 55,155 C70,170 85,160 92,145 C99,130 95,120 80,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='68' y1='130' x2='55' y2='115' stroke='%2381c784' stroke-width='0.25'/%3E%3Cline x1='68' y1='130' x2='65' y2='120' stroke='%2381c784' stroke-width='0.25'/%3E%3Cline x1='68' y1='130' x2='72' y2='125' stroke='%2381c784' stroke-width='0.25'/%3E%3C/g%3E%3Cg opacity='0.04' transform='translate(250,200) rotate(-40)'%3E%3Cpath d='M80,140 C65,115 50,110 42,118 C34,126 40,140 55,155 C70,170 85,160 92,145 C99,130 95,120 80,140 Z' fill='%2366bb6a' stroke='%2381c784' stroke-width='0.4'/%3E%3Cline x1='68' y1='130' x2='55' y2='115' stroke='%2381c784' stroke-width='0.25'/%3E%3Cline x1='68' y1='130' x2='65' y2='120' stroke='%2381c784' stroke-width='0.25'/%3E%3Cline x1='68' y1='130' x2='72' y2='125' stroke='%2381c784' stroke-width='0.25'/%3E%3C/g%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23leaf-pattern-dark-theme-light)'/%3E%3C/svg%3E");
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* App background with detailed realistic leaf pattern */
    [data-testid="stAppViewContainer"] {
        background: var(--bg-primary);
        background-image: 
            url("data:image/svg+xml,%3Csvg width='300' height='300' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='leaf-pattern' x='0' y='0' width='300' height='300' patternUnits='userSpaceOnUse'%3E%3Cg opacity='0.12'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%232e7d32' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.1' transform='translate(140,70) rotate(45)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%232e7d32' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.08' transform='translate(220,180) rotate(-30)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%232e7d32' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.06' transform='translate(40,50) rotate(75)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%232e7d32' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.05' transform='translate(250,50) rotate(120)'%3E%3Cpath d='M70,140 C55,110 40,100 30,110 C20,120 25,135 40,155 C55,175 75,165 85,145 C95,125 90,115 70,140 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.6'/%3E%3Cline x1='57' y1='130' x2='30' y2='110' stroke='%232e7d32' stroke-width='0.4'/%3E%3Cline x1='57' y1='130' x2='50' y2='118' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='55' y2='122' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='62' y2='125' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='57' y1='130' x2='68' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='50' y1='118' x2='45' y2='112' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='50' y1='118' x2='48' y2='115' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='58' y2='120' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='62' y1='125' x2='60' y2='122' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23leaf-pattern)'/%3E%3C/svg%3E");
        background-size: 600px 600px;
        background-repeat: repeat;
    }
    
    /* Main content area with detailed leaf background */
    .main .block-container {
        background: var(--bg-primary);
        background-image: 
            url("data:image/svg+xml,%3Csvg width='400' height='400' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='leaf-pattern-light' x='0' y='0' width='400' height='400' patternUnits='userSpaceOnUse'%3E%3Cg opacity='0.08'%3E%3Cpath d='M90,160 C70,125 50,115 38,125 C26,135 32,155 52,180 C72,205 95,190 105,165 C115,140 110,130 90,160 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.5'/%3E%3Cline x1='74' y1='150' x2='38' y2='125' stroke='%232e7d32' stroke-width='0.35'/%3E%3Cline x1='74' y1='150' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='70' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='80' y2='145' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='85' y2='155' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='92' y2='165' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='65' y1='135' x2='58' y2='128' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='65' y1='135' x2='62' y2='131' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='75' y2='138' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='78' y2='140' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='85' y1='155' x2='82' y2='150' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.06' transform='translate(240,90) rotate(50)'%3E%3Cpath d='M90,160 C70,125 50,115 38,125 C26,135 32,155 52,180 C72,205 95,190 105,165 C115,140 110,130 90,160 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.5'/%3E%3Cline x1='74' y1='150' x2='38' y2='125' stroke='%232e7d32' stroke-width='0.35'/%3E%3Cline x1='74' y1='150' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='70' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='80' y2='145' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='85' y2='155' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='92' y2='165' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='65' y1='135' x2='58' y2='128' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='65' y1='135' x2='62' y2='131' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='75' y2='138' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='78' y2='140' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='85' y1='155' x2='82' y2='150' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3Cg opacity='0.05' transform='translate(300,220) rotate(-40)'%3E%3Cpath d='M90,160 C70,125 50,115 38,125 C26,135 32,155 52,180 C72,205 95,190 105,165 C115,140 110,130 90,160 Z' fill='%234caf50' stroke='%232e7d32' stroke-width='0.5'/%3E%3Cline x1='74' y1='150' x2='38' y2='125' stroke='%232e7d32' stroke-width='0.35'/%3E%3Cline x1='74' y1='150' x2='65' y2='135' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='70' y2='142' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='80' y2='145' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='85' y2='155' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='74' y1='150' x2='92' y2='165' stroke='%232e7d32' stroke-width='0.3'/%3E%3Cline x1='65' y1='135' x2='58' y2='128' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='65' y1='135' x2='62' y2='131' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='75' y2='138' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='80' y1='145' x2='78' y2='140' stroke='%232e7d32' stroke-width='0.2'/%3E%3Cline x1='85' y1='155' x2='82' y2='150' stroke='%232e7d32' stroke-width='0.2'/%3E%3C/g%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23leaf-pattern-light)'/%3E%3C/svg%3E");
        background-size: 700px 700px;
        background-repeat: repeat;
        color: var(--text-primary);
    }
    
    /* Text color overrides */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    
    p, div, span {
        color: var(--text-primary);
    }
    
    /* Streamlit text elements */
    .stMarkdown, .stText, .stCaption {
        color: var(--text-primary);
    }
    
    /* Ensure all text is visible */
    .stSelectbox, .stSlider, .stInfo, .stSuccess, .stWarning, .stError {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar text visibility */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Upload area text */
    .uploadedFile {
        color: var(--text-primary);
    }
    
    /* Progress text */
    .stProgress + div {
        color: var(--text-primary) !important;
    }
    
    /* Custom header styling - Green theme */
    .main-header {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 50%, var(--accent-light) 100%);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px var(--shadow-color);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: #000000;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: #000000;
    }
    
    /* Dark mode header text */
    @media (prefers-color-scheme: dark) {
        .main-header h1 {
            color: #ffffff;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        
        .main-header p {
            color: #ffffff;
        }
    }
    
    [data-theme="dark"] .main-header h1 {
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    [data-theme="dark"] .main-header p {
        color: #ffffff;
    }
    
    /* Card styling */
    .detection-card {
        background: var(--bg-card);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px var(--shadow-color);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        color: var(--text-primary);
    }
    
    .result-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--border-color) 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent-secondary);
        color: var(--text-primary);
    }
    
    .disease-item {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px var(--shadow-color);
        border-left: 4px solid var(--accent-light);
        color: var(--text-primary);
    }
    
    /* Button styling - Green theme */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: white;
			border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
			font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px var(--shadow-color);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow-color);
        background: linear-gradient(135deg, var(--accent-secondary) 0%, var(--accent-light) 100%);
    }
    
    /* Sidebar styling - Green theme */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--border-color) 100%);
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] {
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: var(--text-primary);
    }
    
    /* Upload area styling - Green theme */
    .uploadedFile {
			border-radius: 10px;

        border: 2px dashed var(--accent-secondary);
        background: var(--bg-secondary);
    }
    
    /* Simple upload button styling */
    .stFileUploader > div > div > div {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
			border: none;

        border-radius: 8px;
        color: white;
        padding: 0.75rem 1.5rem;
			font-weight: 600;

        transition: all 0.3s ease;
        box-shadow: 0 4px 15px var(--shadow-color);
        text-align: center;
    }
    
    .stFileUploader > div > div > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow-color);
        background: linear-gradient(135deg, var(--accent-secondary) 0%, var(--accent-light) 100%);
    }
    
    /* Progress bar styling - Green theme */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    }
    
    /* Metric cards - Green theme */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--border-color) 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 3px 10px var(--shadow-color);
        border: 1px solid var(--accent-light);
        color: var(--text-primary);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Don't hide header completely - we need it for sidebar toggle */
    header {visibility: visible;}
    
    /* Make header menu button more visible and accessible */
    header button[aria-label*="menu"],
    header button[aria-label*="Menu"],
    header button[aria-label*="navigation"] {
        visibility: visible !important;
        display: block !important;
        z-index: 1000 !important;
    }
    
    
    /* Force sidebar to be visible when needed */
    [data-testid="stSidebar"] {
        display: flex !important;
        visibility: visible !important;
        position: relative !important;
    }
    
    /* Ensure sidebar has proper width */
    [data-testid="stSidebar"][style*="display: none"] {
        display: flex !important;
    }
    
    /* Ensure sidebar is not hidden by default */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            transform: translateX(0) !important;
        }
    }
    
    /* Make sure main content adjusts for sidebar */
    [data-testid="stAppViewContainer"] > div {
        transition: margin-left 0.3s ease;
    }
	</style>


<script>
// Auto-detect browser theme preference
function detectTheme() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        document.documentElement.setAttribute('data-theme', 'light');
    }
}

// Run on page load
detectTheme();

// Listen for theme changes
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectTheme);
}

// Auto-scroll helper functions (simplified)
function scrollToElement(elementId, fallbackSelector, block = 'start') {
    try {
        const element = document.getElementById(elementId) || document.querySelector(fallbackSelector);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: block });
            return true;
        }
    } catch(e) {
        console.error('Scroll error:', e);
    }
    return false;
}

</script>
""", unsafe_allow_html=True)

# Model configuration
BASE_MODEL_PATH = "BaseModel.pt"
ENHANCED_MODEL_PATH = "EnhancedModel.pt"

# UI component keys
FILE_UPLOADER_KEY = "calamansi_image_uploader"
RETAKE_PANEL_KEY = "show_retake_panel"
RETAKE_UPLOADER_KEY = "retake_image_uploader"
RETAKE_IMAGE_BYTES_KEY = "retake_image_bytes"
RETAKE_IMAGE_NAME_KEY = "retake_image_name"
AUTO_DETECT_KEY = "auto_detect_retake"



@st.cache_resource
def load_detector(model_path: str):
    return YoloDiseaseDetector(model_path=model_path)

def get_disease_info(disease_name):
    """
    Get disease description and recommendations.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        dict: Dictionary containing 'description' and 'recommendation' keys
    """
    disease_info = {
        "Black Spot": {
            "description": "Fungal disease causing dark, circular spots on leaves and fruits, often with yellow halos.",
            "recommendation": "Remove infected leaves and fruits, apply fungicides containing copper or mancozeb, improve air circulation, avoid overhead watering, and maintain proper spacing between plants."
        },
        "Citrus Greening": {
            "description": "Serious bacterial disease transmitted by psyllids, causes yellowing and stunted growth.",
            "recommendation": "Remove infected trees, control psyllid vectors, use disease-free planting material."
        },
        "Citrus Scab": {
            "description": "Fungal disease causing corky, raised lesions on fruits and leaves.",
            "recommendation": "Apply fungicides during wet periods, prune for better air circulation."
        },
        "Healthy": {
            "description": "Plant appears healthy with no visible disease symptoms.",
            "recommendation": "Continue regular care and monitoring for early disease detection."
        },
        "Mites Infestation": {
            "description": "Tiny arachnids feeding on plant sap, causing stippling and discoloration.",
            "recommendation": "Apply miticides, increase humidity, remove heavily infested leaves."
        },
        "Powdery Mildew": {
            "description": "Fungal disease causing white powdery coating on leaves and stems.",
            "recommendation": "Improve air circulation, apply fungicides, avoid overhead watering."
        },
        "Scales Infestation": {
            "description": "Small insects that attach to plant surfaces and feed on sap.",
            "recommendation": "Apply horticultural oil, remove heavily infested branches, encourage natural predators."
        },
        "Tristeza": {
            "description": "Viral disease causing decline, yellowing, and stunting of citrus trees.",
            "recommendation": "Remove infected trees, use virus-free rootstocks, control aphid vectors."
        },
        "Xyloporosis": {
            "description": "Viral disease causing wood pitting and decline in citrus trees.",
            "recommendation": "Remove infected trees, use certified disease-free planting material."
        }
    }
    
    return disease_info.get(disease_name, {
        "description": "Disease information not available.",
        "recommendation": "Consult with a plant pathologist for specific treatment recommendations."
    })


def get_class_severity(avg_conf):
    """Map average confidence to severity label and color"""
    if avg_conf >= 0.75:
        return "Severe", "#e53935"
    if avg_conf >= 0.5:
        return "Elevated", "#fb8c00"
    return "Mild", "#8bc34a"


def render_retake_panel():
    """Display the retake uploader interface."""
    st.markdown("##### Retake Image")
    st.caption("Drag & drop or browse another image to analyze.")
    retake_uploader = st.file_uploader(
        "Upload Replacement Image",
        key=RETAKE_UPLOADER_KEY,
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
        help="Upload a new image without leaving this section"
    )
    action_cols = st.columns(2)
    with action_cols[0]:
        retake_detect = st.button("⚡ Detect Retake", key="retake_detect_btn", use_container_width=True)
    with action_cols[1]:
        retake_cancel = st.button("✖️ Cancel", key="retake_cancel_btn", use_container_width=True)
    if retake_cancel:
        st.session_state[RETAKE_PANEL_KEY] = False
        st.session_state.pop(RETAKE_UPLOADER_KEY, None)
    if retake_detect:
        if retake_uploader is None:
            st.warning("Please upload an image before running detection.")
        else:
            st.session_state[RETAKE_IMAGE_BYTES_KEY] = retake_uploader.getvalue()
            st.session_state[RETAKE_IMAGE_NAME_KEY] = retake_uploader.name
            st.session_state[AUTO_DETECT_KEY] = True
            st.session_state[RETAKE_PANEL_KEY] = False
            st.session_state.pop(RETAKE_UPLOADER_KEY, None)
            st.rerun()


def build_detection_analytics(detections):
    """
    Compute aggregate analytics for detections.
    
    Severity computation rules (based on count):
    - Single target/detection:
      - Diseased → severity = 1.0 (100%)
      - Healthy → severity = 0
    - Multiple targets/detection:
      - Severity = (# diseased targets) / (total # of targets)
      - Reflects proportion of disease in the image or sample
    
    Args:
        detections (list): List of detection dictionaries with 'class_name' and 'confidence'
        
    Returns:
        dict: Analytics dictionary with severity, counts, and statistics
    """
    if not detections:
        return {
            "total_detections": 0,
            "unique_classes": 0,
            "class_stats": [],
            "severity_score": 0,
            "severity_label": "None",
            "severity_color": "#66bb6a",
            "summary_text": "No diseases detected. Plant appears healthy.",
            "top_class": None,
            "weighted_conf": 0,
            "detection_density": 0,
            "diseased_count": 0,
            "healthy_count": 0
        }
    
    class_map = {}
    for detection in detections:
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        if class_name not in class_map:
            class_map[class_name] = []
        class_map[class_name].append(confidence)
    
    class_stats = []
    total_detections = len(detections)
    
    # Count diseased and healthy detections
    diseased_count = 0
    healthy_count = 0
    
    for class_name, conf_list in class_map.items():
        count = len(conf_list)
        avg_conf = sum(conf_list) / count
        max_conf = max(conf_list)
        
        class_stats.append({
            "class_name": class_name,
            "count": count,
            "avg_conf": avg_conf,
            "max_conf": max_conf
        })
        
        # Count healthy vs diseased
        if class_name.lower() == "healthy":
            healthy_count += count
        else:
            diseased_count += count
    
    # Compute severity based on count rules
    total_targets = diseased_count + healthy_count
    
    if total_targets == 0:
        severity_score = 0.0
        severity_label = "None"
        severity_color = "#66bb6a"
        summary_text = "No detections found."
    elif total_targets == 1:
        # Single target/detection
        if diseased_count == 1:
            # Single diseased → severity = 1.0
            severity_score = 1.0
            severity_label = "High"
            severity_color = "#e53935"
            summary_text = "Single diseased target detected. Severity: 100%."
        else:
            # Single healthy → severity = 0
            severity_score = 0.0
            severity_label = "None"
            severity_color = "#66bb6a"
            summary_text = "Single healthy target detected. No disease severity."
    else:
        # Multiple targets → severity = (# diseased) / (total # of targets)
        severity_score = diseased_count / total_targets
        severity_label = "High" if severity_score >= 0.7 else "Moderate" if severity_score >= 0.4 else "Low"
        severity_color = "#e53935" if severity_score >= 0.7 else "#fb8c00" if severity_score >= 0.4 else "#8bc34a"
        summary_text = f"{diseased_count} diseased and {healthy_count} healthy targets. Severity: {severity_score * 100:.1f}% ({diseased_count}/{total_targets})."
    
    # Calculate weighted confidence for display (average of all confidences)
    all_confidences = [conf for conf_list in class_map.values() for conf in conf_list]
    weighted_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # Detection density for reference
    detection_density = min(1.0, total_detections / 5)
    
    # Identify the class with highest risk (by count)
    top_class = max(
        class_stats,
        key=lambda item: item["count"],
        default=None
    )
    
    if top_class and top_class["class_name"].lower() != "healthy":
        summary_text = (
            f"Most prominent issue: {top_class['class_name']} "
            f"({top_class['count']} detections). "
            f"Overall severity: {severity_score * 100:.1f}% ({diseased_count}/{total_targets} diseased)."
        )
    
    return {
        "total_detections": total_detections,
        "unique_classes": len(class_map),
        "class_stats": class_stats,
        "severity_score": severity_score,
        "severity_label": severity_label,
        "severity_color": severity_color,
        "summary_text": summary_text,
        "top_class": top_class,
        "weighted_conf": weighted_conf,
        "detection_density": detection_density,
        "diseased_count": diseased_count,
        "healthy_count": healthy_count
    }

def main():
    # Initialize session state for scroll tracking
    if 'scroll_to_results' not in st.session_state:
        st.session_state.scroll_to_results = False
    
    # JavaScript to ensure sidebar is visible and force show on page load
    st.markdown("""
    <script>
    (function() {
        function forceShowSidebar() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                const rect = sidebar.getBoundingClientRect();
                const isVisible = rect.width > 100;
                
                if (!isVisible) {
                    // Force sidebar to be visible with proper width
                    sidebar.style.cssText = `
                        position: relative !important;
                        display: flex !important;
                        visibility: visible !important;
                        opacity: 1 !important;
                        transform: translateX(0) !important;
                        width: 21rem !important;
                        min-width: 21rem !important;
                        max-width: 21rem !important;
                    `;
                    
                    // Adjust main content to make room
                    const mainBlock = document.querySelector('[data-testid="stAppViewContainer"] > div');
                    if (mainBlock) {
                        const currentMargin = window.getComputedStyle(mainBlock).marginLeft;
                        if (currentMargin === '0px' || !currentMargin) {
                            mainBlock.style.marginLeft = '21rem';
                        }
                    }
                }
            }
        }
        
        // Try multiple times to catch sidebar after it loads
        forceShowSidebar();
        setTimeout(forceShowSidebar, 50);
        setTimeout(forceShowSidebar, 100);
        setTimeout(forceShowSidebar, 300);
        setTimeout(forceShowSidebar, 500);
        setTimeout(forceShowSidebar, 1000);
        
        // Also watch for DOM changes
        const observer = new MutationObserver(function(mutations) {
            forceShowSidebar();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['style', 'class']
        });
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # Modern header
    st.markdown("""

    <div class="main-header">
        <h1>🍃 Calamansi Disease Detector</h1>
        <p>AI-Powered Plant Health Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with modern design
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        model_choice = st.selectbox(
            "🤖 Select AI Model:",
            ["Enhanced Model (Recommended)", "Base Model"],
            help="Enhanced model provides better accuracy for disease detection"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Detection Settings")
        
        confidence = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Higher values = more strict detection"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Supported Diseases:**

        • Black Spot
        • Citrus Greening  
        • Citrus Scab
        • Healthy 
        • Scales Infestation
        """)
    
    # Main content area - Single column layout
    st.markdown("### 📸 Upload Image")
    
    # Simple upload button
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Upload a clear image of calamansi leaves or fruits",
        label_visibility="collapsed",
        key=FILE_UPLOADER_KEY
    )
    
    # Track if we need to scroll to detect button (when file is newly uploaded)
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    # Check if a new file was uploaded
    scroll_to_detect = False
    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            scroll_to_detect = True
            st.session_state.last_uploaded_file = uploaded_file.name
    else:
        st.session_state.last_uploaded_file = None
    auto_detect = st.session_state.pop(AUTO_DETECT_KEY, False)
    pending_bytes = st.session_state.pop(RETAKE_IMAGE_BYTES_KEY, None)
    pending_name = st.session_state.pop(RETAKE_IMAGE_NAME_KEY, None)
    if pending_bytes is not None:
        uploaded_file = io.BytesIO(pending_bytes)
        uploaded_file.name = pending_name or "retake-image.png"
        auto_detect = True
        st.session_state[RETAKE_PANEL_KEY] = False
    
    if st.session_state.get(RETAKE_PANEL_KEY):
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        render_retake_panel()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display uploaded image
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            
            # Auto-scroll to detect button when image is uploaded
            if scroll_to_detect:
                st.markdown("""
                <script>
                (function() {
                    function scroll() {
                        const target = document.getElementById('detect-button-area');
                        if (target) {
                            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            return;
                        }
                        // Fallback: find button
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const detectBtn = buttons.find(b => b.textContent.includes('Detect Disease'));
                        if (detectBtn) detectBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    setTimeout(scroll, 300);
                    setTimeout(scroll, 800);
                })();
                </script>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="detection-card">', unsafe_allow_html=True)
            st.markdown("#### 📷 Uploaded Image")
            # Center the uploaded image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div style="display: flex; justify-content: center; align-items: center;">', unsafe_allow_html=True)
                st.image(image, width=400)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)



            # Detect button - centered under the image
            st.markdown('<div id="detect-button-area"></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.write("")  # Empty column for spacing
            
            with col2:
                detect_clicked = st.button("🔍 Detect Disease", help="Analyze the uploaded image", use_container_width=True, key="detect_disease_btn")
                
                # Set flag to scroll to results when detect is clicked
                if detect_clicked:
                    st.session_state.scroll_to_results = True
            
            with col3:
                st.write("")  # Empty column for spacing
            
            # Run detection when button is clicked
            if detect_clicked or auto_detect:
                # Add marker div for scrolling
                st.markdown('<div id="analysis-results-start"></div>', unsafe_allow_html=True)
                st.markdown("### 🔍 Analysis Results")
                
                # Select model path
                model_path = ENHANCED_MODEL_PATH if "Enhanced" in model_choice else BASE_MODEL_PATH
                
                # Run detection with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Initializing AI model...")
                progress_bar.progress(20)
                
                status_text.text("🔍 Analyzing image...")
                progress_bar.progress(60)
                
                try:
                    detector = load_detector(model_path)
                    annotated_image, detections = detector.predict_image(
                        image,
                        conf=confidence,
                        iou=0.50,
                        imgsz=640
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Build analytics first
                    analytics = build_detection_analytics(detections)
                    
                    # Scroll to results after completion
                    if st.session_state.get('scroll_to_results', False):
                        st.markdown("""
                        <script>
                        (function() {
                            function scroll() {
                                const target = document.getElementById('analysis-results-start') ||
                                              document.querySelector('.result-card') ||
                                              Array.from(document.querySelectorAll('h3')).find(h => 
                                                  h.textContent.includes('Analysis Results') || 
                                                  h.textContent.includes('Advanced Analytics')
                                              );
                                if (target) {
                                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                } else {
                                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                                }
                            }
                            setTimeout(scroll, 500);
                            setTimeout(scroll, 1200);
                        })();
                        </script>
                        """, unsafe_allow_html=True)
                        st.session_state.scroll_to_results = False
                    
                    # Display Advanced Analytics at the top (where total detection and class were)
                    st.markdown("### 📈 Advanced Analytics")
                    severity_percent = analytics['severity_score'] * 100
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>Overall Severity: <span style="color: {analytics['severity_color']};">{severity_percent:.1f}% ({analytics['severity_label']})</span></h4>
                        <p>{analytics['summary_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display average confidence per class
                    num_classes = len(analytics['class_stats'])
                    
                    if num_classes > 0:
                        # Always show per-class average confidence
                        st.markdown("#### **Per-Class Average Confidence**")
                        
                        # Create columns for class confidences (max 4 per row for readability)
                        max_cols_per_row = 4
                        class_stats = analytics['class_stats']
                        
                        # Display classes in rows of max_cols_per_row
                        for row_start in range(0, num_classes, max_cols_per_row):
                            row_end = min(row_start + max_cols_per_row, num_classes)
                            row_classes = class_stats[row_start:row_end]
                            class_cols = st.columns(len(row_classes))
                            
                            for idx, class_stat in enumerate(row_classes):
                                with class_cols[idx]:
                                    # Truncate long class names for display
                                    display_name = class_stat['class_name']
                                    if len(display_name) > 15:
                                        display_name = display_name[:12] + "..."
                                    st.metric(f"{display_name}", f"{class_stat['avg_conf'] * 100:.1f}%")
                        
                        if num_classes > max_cols_per_row:
                            st.caption(f"Showing all {num_classes} classes. See details table below for more information.")
                        
                        # Additional metrics row
                        total_targets = analytics.get('diseased_count', 0) + analytics.get('healthy_count', 0)
                        severity_percent = analytics['severity_score'] * 100
                        severity_display = f"{severity_percent:.1f}%"
                        if total_targets > 0:
                            severity_display += f" [{analytics.get('diseased_count', 0)}/{total_targets}]"
                        
                        metric_cols = st.columns(5)
                        with metric_cols[0]:
                            st.metric("Severity Score", severity_display)
                        with metric_cols[1]:
                            st.metric("Total Count", f"{total_targets}")
                        with metric_cols[2]:
                            st.metric("Diseased Count", f"{analytics.get('diseased_count', 0)}")
                        with metric_cols[3]:
                            st.metric("Healthy Count", f"{analytics.get('healthy_count', 0)}")
                        with metric_cols[4]:
                            st.metric("Unique Classes", f"{analytics['unique_classes']}")
                    else:
                        # Fallback if no class stats
                        total_targets = analytics.get('diseased_count', 0) + analytics.get('healthy_count', 0)
                        severity_percent = analytics['severity_score'] * 100
                        severity_display = f"{severity_percent:.1f}%"
                        if total_targets > 0:
                            severity_display += f" [{analytics.get('diseased_count', 0)}/{total_targets}]"
                        
                        metric_cols = st.columns(5)
                        with metric_cols[0]:
                            st.metric("Severity Score", severity_display)
                        with metric_cols[1]:
                            st.metric("Total Count", f"{total_targets}")
                        with metric_cols[2]:
                            st.metric("Diseased Count", f"{analytics.get('diseased_count', 0)}")
                        with metric_cols[3]:
                            st.metric("Healthy Count", f"{analytics.get('healthy_count', 0)}")
                        with metric_cols[4]:
                            st.metric("Unique Classes", f"{analytics['unique_classes']}")
                    
                    # Show severity computation explanation
                    with st.expander("🔬 How Severity is Computed", expanded=False):
                        st.markdown("""
                        #### **Severity Computation Rules (Based on Count):**
                        
                        **Single target/detection:**
                        - Diseased → `severity = 100%`
                        - Healthy → `severity = 0%`
                        
                        **Multiple targets/detection:**
                        - `severity = (# diseased targets) / (total # of targets)`
                        - Reflects proportion of disease in the image or sample
                        
                        ---
                        
                        #### **Example Outputs:**
                        
                        | Detections | Severity |
                        |------------|----------|
                        | 1 diseased | 100% |
                        | 1 healthy | 0% |
                        | 6 diseased + 1 healthy | 6/7 ≈ 85.7% |
                        | 3 diseased + 2 healthy | 3/5 = 60% |
                        
                        ---
                        
                        #### **Current Calculation:**
                        """)
                        
                        # Get counts from analytics
                        diseased_count = analytics.get('diseased_count', 0)
                        healthy_count = analytics.get('healthy_count', 0)
                        total_targets = diseased_count + healthy_count
                        
                        # Determine which rule applies
                        if total_targets == 0:
                            rule_text = "**No detections found**"
                            formula_text = f"severity = 0%"
                        elif total_targets == 1:
                            if diseased_count == 1:
                                rule_text = "**Single target: Diseased** → severity = 100%"
                                formula_text = f"severity = 100%"
                            else:
                                rule_text = "**Single target: Healthy** → severity = 0%"
                                formula_text = f"severity = 0%"
                        else:
                            rule_text = "**Multiple targets** → severity = (# diseased) / (total)"
                            severity_percent = analytics['severity_score'] * 100
                            formula_text = f"severity = {diseased_count} / {total_targets} = {severity_percent:.1f}%"
                        
                        severity_percent = analytics['severity_score'] * 100
                        st.markdown(f"""
                        {rule_text}
                        
                        **Formula**: {formula_text}
                        
                        **Result**: Severity Score = **{severity_percent:.1f}%** ({analytics['severity_label']})
                        
                        **Breakdown**:
                        - Diseased targets: {diseased_count}
                        - Healthy targets: {healthy_count}
                        - Total targets: {total_targets}
                        - Severity: {diseased_count}/{total_targets} = {severity_percent:.1f}%
                        """)
                    
                    # Display annotated image and results in two columns
                    st.markdown("### 🔍 Detection Results")
                    
                    # Two column layout for results
                    result_col1, result_col2 = st.columns([1, 1])
                    
                    with result_col1:
                        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
                        st.markdown("#### 🎯 Annotated Image")
                        st.image(annotated_image, width=400)
                        retake_btn_cols = st.columns([0.4, 0.6])
                        with retake_btn_cols[0]:
                            if st.button("Retake", key="retake_trigger_btn"):
                                st.session_state[RETAKE_PANEL_KEY] = True
                                st.session_state.pop(FILE_UPLOADER_KEY, None)
                                st.session_state.pop(RETAKE_UPLOADER_KEY, None)
                                st.session_state.pop(RETAKE_IMAGE_BYTES_KEY, None)
                                st.session_state.pop(RETAKE_IMAGE_NAME_KEY, None)
                                st.rerun()
                        with retake_btn_cols[1]:
                            st.write("")
                        st.markdown('</div>', unsafe_allow_html=True)
                        

                    with result_col2:
                        # Display detection results
                        if len(detections) == 0:

                            st.markdown("""
                            <div class="result-card">
                                <h4>✅ No Diseases Detected</h4>
                                <p>The plant appears to be healthy or the image may be unclear.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:

                            # Group detections by class name
                            class_groups = {}
                            for detection in detections:
                                class_name = detection['class_name']
                                if class_name not in class_groups:
                                    class_groups[class_name] = []
                                class_groups[class_name].append(detection['confidence'])
                            
                            # Display each unique class WITHOUT confidence level
                            for i, (class_name, confidences) in enumerate(class_groups.items(), 1):
                                # Get disease description and recommendations
                                disease_info = get_disease_info(class_name)
                                
                                st.markdown(f"""
                                <div class="disease-item">
                                    <h5>🔬 Class {i}</h5>
                                    <p><strong>Class Name:</strong> {class_name}</p>
                                    <p><strong>Description:</strong> {disease_info['description']}</p>
                                    <p><strong>Recommendation:</strong> {disease_info['recommendation']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display detailed table of all detections
                            st.markdown("### 📊 Detection Details Table")
                            
                            # Group detections by class for the new table format
                            class_detections = {}
                            for detection in detections:
                                class_name = detection['class_name']
                                if class_name not in class_detections:
                                    class_detections[class_name] = []
                                class_detections[class_name].append(detection['confidence'])
                            
                            # Create the new table format
                            if class_detections:
                                # Find the maximum number of detections for any class
                                max_detections = max(len(detections) for detections in class_detections.values())
                                
                                # Create table data in the new format
                                table_data = []
                                class_names = list(class_detections.keys())
                                
                                for row in range(max_detections):
                                    row_data = {"Count": row + 1}
                                    
                                    # Add each class column
                                    for i, class_name in enumerate(class_names):
                                        class_key = f"Class {i + 1}"
                                        if row < len(class_detections[class_name]):
                                            confidence = class_detections[class_name][row]
                                            row_data[class_key] = f"{class_name} ({confidence:.2f})"
                                        else:
                                            row_data[class_key] = ""
                                    
                                    table_data.append(row_data)
                                
                                # Display the new table
                                st.table(table_data)
                
                except Exception as e:
                    st.error(f"❌ Error during detection: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
                    import traceback
                    st.exception(e)
        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            import traceback
            st.exception(e)


    # Footer with modern design
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🌿 Plant Health</h4>
            <p>AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 YOLO Technology</h4>
            <p>Advanced deep learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📱 Modern Interface</h4>
            <p>User-friendly design</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


