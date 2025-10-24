# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# FILE: app.py
# OBIETTIVO: Eseguire la dashboard Dash caricando i modelli da ./Models_APP/
# ESEGUIRE QUESTO SCRIPT PER LANCIARE L'APPLICAZIONE.
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import torch.nn as nn
import joblib
import os

# Inizializzazione app Dash (solo UNA volta)
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['/assets/style.css'])
server = app.server
app.title = "Pipe Predictions"


# ==============================================================================
# DEFINIZIONE CLASSE MLP
# ==============================================================================

# Necessario per caricare lo state_dict PyTorch
class ProfileMLP(nn.Module):
    def __init__(self, input_features):
        super(ProfileMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)



# ==============================================================================
# CARICAMENTO MODELLI PRE-ADDESTRATI
# ==============================================================================

load_dir = "Models_APP"
print(f"Caricamento dei modelli e degli scaler da '{load_dir}'...")

try:
    # Caricamento Artefatti GPR DR
    gaussian_process_dr = joblib.load(os.path.join(load_dir, 'gpr_model_dr.joblib'))
    scaler_scalari_gpr = joblib.load(os.path.join(load_dir, 'scaler_gpr_input.joblib'))
    scaler_output_dr = joblib.load(os.path.join(load_dir, 'scaler_gpr_output_dr.joblib'))
    params_gpr_plot_axis = joblib.load(os.path.join(load_dir, 'gpr_plot_params.joblib'))

    # Caricamento Artefatti GPR AR 
    gaussian_process_ar = joblib.load(os.path.join(load_dir, 'gpr_model_ar.joblib'))
    scaler_output_ar = joblib.load(os.path.join(load_dir, 'scaler_gpr_output_ar.joblib'))

    # Caricamento Artefatti MLP AP
    scaler_mlp = joblib.load(os.path.join(load_dir, 'scaler_mlp_ap.joblib'))
    pipe_abs = joblib.load(os.path.join(load_dir, 'pipe_abs_ap.joblib')) 

    # Caricamento Modello MLP AP
    input_features_count = scaler_mlp.n_features_in_
    model_mlp = ProfileMLP(input_features=input_features_count)
    model_mlp.load_state_dict(torch.load(os.path.join(load_dir, 'mlp_model_ap.pth')))
    model_mlp.eval()

    # Caricamento Artefatti Comuni
    bounds_curv = joblib.load(os.path.join(load_dir, 'bounds_curv.joblib'))

except FileNotFoundError as e:
    print("="*50)
    print(f"ERRORE: File dei modelli non trovati nella cartella '{load_dir}'. Dettaglio: {e}")
    print("Assicurarsi di aver eseguito 'train_models.py' aggiornato almeno una volta.")
    print("="*50)
    exit()

print("Caricamento completato. Avvio del server Dash...")

# ==============================================================================
# DEFINIZIONI PER LAYOUT
# ==============================================================================

# Limiti statici 
bounds = {
    "D_out": (32, 42),
    "Wall_Thickness": (25, 35),
    "Compression_Force": (1e6, 5e6),
    "External_Pressure": (0.5, 2),
    "Young_Modulus": (190000, 230000),
    "SY1": (350, 400),
    "SY2": (450, 500),
    "EP2": (0.1, 0.2),
    "Ovalisation": (0, 0.03)
}

# Marks per slider Pressione Esterna (usato sotto)
marks_ep = {
    v: f"{v}".rstrip("0").rstrip(".")
    for v in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
}

# --- Stili CSS Centralizzati ---
# Stile per il contenitore principale (grigio chiaro)
STYLE_MAIN_CONTAINER = {
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', # Stack font moderni
    'backgroundColor': '#e9ecef', # Sfondo leggermente più scuro (precedente: #f0f2f5)
    'padding': '20px', # Aumentato padding generale
    'minHeight': '100vh' # Assicura che lo sfondo copra l'intera altezza
}

# Stile per la sidebar (contenitore delle card)
STYLE_SIDEBAR = {
    'width': '28%', 
    'verticalAlign': 'top',
    'boxSizing': 'border-box',
    'marginRight': '20px', 
    # Le proprietà 'padding', 'backgroundColor', 'borderRadius', 'boxShadow', 'border' sono state rimosse
}

# Stile per l'area dei grafici
STYLE_CONTENT = {
    'width': '72%',
    'verticalAlign': 'top',
    'boxSizing': 'border-box'
}

# Stile per una singola "card" che contiene un grafico
STYLE_GRAPH_CARD = {
    'backgroundColor': 'white',
    'borderRadius': '8px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.07)', # Ombra più morbida
    'padding': '10px', # Padding interno ridotto (era 20px)
    'marginBottom': '20px', # Spazio tra le card
    'border': '1px solid #dee2e6' # Bordo sottile
}

# NUOVO STILE: Card per i gruppi di controlli nella sidebar
STYLE_CONTROL_CARD = {
    'padding': '25px',
    'backgroundColor': 'white',
    'borderRadius': '8px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.07)',
    'border': '1px solid #dee2e6'
    # Il margine inferiore verrà applicato inline
}

# Stile per un gruppo di controlli
STYLE_CONTROL_GROUP = {
    'marginBottom': '25px' # Spazio sotto ogni gruppo di slider
}

# Stile per i titoli H5 dei gruppi di controllo
STYLE_H5 = {
    'marginTop': '0px',
    'marginBottom': '15px',
    'paddingBottom': '10px',
    'borderBottom': '1px solid #eee', # Linea divisoria
    'color': '#495057' # Colore testo scuro ma non nero
}

# ==============================================================================
# LAYOUT DELL'APP DASH
# ==============================================================================


app.layout = html.Div([
    
    # Titolo spostato fuori dalla colonna, centrato
    html.H1("Pipe Performance Analysis: Area Profile (AP), Area Reduction (AR), Diameter Reduction (DR)", 
            style={'textAlign': 'center', 'marginBottom': '35px', 'color': '#333'}),
    
    html.Div([
        # Colonna Sinistra (Controlli)
        html.Div([

            # --- CASELLA 1: Parametri Analisi e Thresholds ---
            html.Div([
                html.Div([
                    html.H5("Display Thresholds", style=STYLE_H5),
                    html.Label("Threshold for AR (%)"),
                    dcc.Slider(id='slider-ar-line', min=0, max=20, step=0.25, value=5,
                               marks={v: f'{v:.1f}' for v in np.linspace(0, 15, 7)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label("Threshold for DR (%)", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-dr-line', min=0, max=20, step=0.25, value=0,
                               marks={v: f'{v:.1f}' for v in np.linspace(0, 15, 7)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], style=STYLE_CONTROL_GROUP),

                html.Div([
                    html.H5("Analysis Parameters", style=STYLE_H5),
                    html.Label(f"CURVATURE (rad) - (Area Profile Graph Only)", style={'marginTop': '15px'}), 
                    dcc.Slider(id='slider-curv',
                               min=bounds_curv[0], max=bounds_curv[1], step=0.005, value=np.max(bounds_curv),
                               marks={v: f"{v:.3f}" for v in np.linspace(bounds_curv[0], bounds_curv[1], 10)}, 
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], style=STYLE_CONTROL_GROUP),
            
            # Applica lo stile della card e un margine inferiore
            ], style={**STYLE_CONTROL_CARD, 'marginBottom': '20px'}), 

            # --- CASELLA 2: Parametri Geometrici, Carico, Materiali ---
            html.Div([
                html.Div([
                    html.H5("Geometric Parameters", style=STYLE_H5),
                    html.Label("OUTER DIAMETER (inches)"),
                    dcc.Slider(id='slider-od', min=bounds['D_out'][0], max=bounds['D_out'][1], step=0.5, value=40,
                               marks={i: str(i) for i in range(bounds['D_out'][0], bounds['D_out'][1] + 1, 1)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label("WALL THICKNESS (mm)", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-wt', min=bounds['Wall_Thickness'][0], max=bounds['Wall_Thickness'][1], step=0.5, value=27,
                               marks={i: str(i) for i in range(bounds['Wall_Thickness'][0], bounds['Wall_Thickness'][1] + 1, 1)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label(f"OVALISATION", style={'marginTop': '15px'}), 
                    dcc.Slider(id='slider-oval', min=bounds['Ovalisation'][0], max=bounds['Ovalisation'][1], step=0.001, value=0,
                               marks={v: f'{v:.3f}' for v in np.arange(bounds['Ovalisation'][0], bounds['Ovalisation'][1] + 0.0001, 0.005)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], style=STYLE_CONTROL_GROUP),

                html.Div([
                    html.H5("Load Conditions", style=STYLE_H5),
                    html.Label("AXIAL LOAD (kN)"),
                    dcc.Slider(id='slider-al', min=bounds['Compression_Force'][0], max=bounds['Compression_Force'][1], step=5e5, value=2e6,
                               marks={int(i): f'{i/1e6}e6' for i in np.arange(bounds['Compression_Force'][0], bounds['Compression_Force'][1] + 1, 5e5)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label("EXTERNAL PRESSURE (MPa)", style={'marginTop': '15px'}),
                    dcc.Slider(
                        id="slider-ep", min=0.5, max=2.0, step=0.1, value=1,
                        marks=marks_ep, # <-- USA LA VARIABILE DEFINITA
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], style=STYLE_CONTROL_GROUP),

                html.Div([
                    html.H5("Material Properties", style=STYLE_H5),
                    html.Label(f"YOUNG MODULUS (MPa)"),
                    dcc.Slider(id='slider-ym', min=bounds['Young_Modulus'][0], max=bounds['Young_Modulus'][1], step=1000, value=210000,
                               marks={i: f'{int(i/1000)}k' for i in range(bounds['Young_Modulus'][0], bounds['Young_Modulus'][1] + 1, 10000)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label(f"YIELD STRESS - SY1 (MPa)", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-sy1', min=bounds['SY1'][0], max=bounds['SY1'][1], step=5, value=370,
                               marks={i: str(i) for i in range(bounds['SY1'][0], bounds['SY1'][1] + 1, 10)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label(f"LIMIT STRESS - SY2 (MPa)", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-sy2', min=bounds['SY2'][0], max=bounds['SY2'][1], step=5, value=470,
                               marks={i: str(i) for i in range(bounds['SY2'][0], bounds['SY2'][1] + 1, 10)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label(f"EQUIVALENT PLASTIC MODULUS (EP2)", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-ep2', min=bounds['EP2'][0], max=bounds['EP2'][1], step=0.01, value=0.1,
                               marks={v: f'{v:.2f}' for v in np.arange(bounds['EP2'][0], bounds['EP2'][1] + 0.001, 0.02)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], style=STYLE_CONTROL_GROUP),
            
            # Applica solo lo stile della card (senza margine)
            ], style=STYLE_CONTROL_CARD), 
            
        ], style=STYLE_SIDEBAR), # Questa Div usa il NUOVO STYLE_SIDEBAR (solo contenitore)
        
        # Colonna Destra (Grafici)
        html.Div([
            # AP grande in una card
            html.Div(
                dcc.Graph(id='ap-prediction-graph', style={'height': '430px', 'width': '100%'}),
                style=STYLE_GRAPH_CARD
            ),
            
            # AR e DR affiancati
            html.Div([
                html.Div(
                    dcc.Graph(id='ar-prediction-graph', style={'height': '380px', 'width': '100%'}),
                    style={**STYLE_GRAPH_CARD, 'width': '49.5%', 'marginBottom': '0'} # Card
                ),
                html.Div(
                    dcc.Graph(id='dr-prediction-graph', style={'height': '380px', 'width': '100%'}),
                    style={**STYLE_GRAPH_CARD, 'width': '49.5%', 'marginBottom': '0'} # Card
                ),
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style=STYLE_CONTENT),

    ], style={'width': '100%', 'display': 'flex'})
], style=STYLE_MAIN_CONTAINER)


# ==============================================================================
# 3. FUNZIONE HELPER PER GRAFICI GPR
# ==============================================================================

def create_gpr_figure(model_gpr, scaler_out, test_scalari_norm, x_points,
                      threshold_value, title, y_title, line_color,
                      fill_color, line_name, points_name, ci_name,
                      threshold_prefix):
    """
    Funzione helper per generare una figura GPR (DR o AR).
    Esegue predizione, inverse transform, PCHIP e plotting.
    """
    
    # 1. Predizione GPR
    mean_pred_scaled, std_pred_scaled = model_gpr.predict(test_scalari_norm, return_std=True)
    
    # 2. Inverse scaling + inverse log transform
    mean_pred_log = scaler_out.inverse_transform(mean_pred_scaled).flatten()
    std_pred_log = (std_pred_scaled * scaler_out.scale_).flatten()
    
    # 3. Media e limiti CI nello spazio originale (assumendo log-normale)
    mean_prediction = np.exp(mean_pred_log + 0.5 * std_pred_log**2) - 1
    lower_bound = np.exp(mean_pred_log - 1.96 * std_pred_log) - 1
    upper_bound = np.exp(mean_pred_log + 1.96 * std_pred_log) - 1
    lower_bound = np.maximum(lower_bound, 0) # Assicura non negatività

    # 4. Interpolazione PCHIP per curve lisce
    x_smooth_gpr = np.linspace(x_points.min(), x_points.max(), 300)
    
    pchip_mean = PchipInterpolator(x_points, mean_prediction)
    y_smooth_mean = pchip_mean(x_smooth_gpr)
    
    pchip_lower = PchipInterpolator(x_points, lower_bound)
    y_smooth_lower = pchip_lower(x_smooth_gpr)
    
    pchip_upper = PchipInterpolator(x_points, upper_bound)
    y_smooth_upper = pchip_upper(x_smooth_gpr)
    
    y_smooth_lower = np.maximum(y_smooth_lower, 0) # Assicura non negatività dopo interpolazione

    # 5. Creazione figura Plotly
    fig = go.Figure()
    
    # Confidence Interval (Area)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_smooth_gpr, x_smooth_gpr[::-1]]),
        y=np.concatenate([y_smooth_upper, y_smooth_lower[::-1]]),
        fill='toself', fillcolor=fill_color,
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="none", name=ci_name
    ))
    
    # Linea Media (PCHIP)
    fig.add_trace(go.Scatter(
        x=x_smooth_gpr, y=y_smooth_mean, mode='lines',
        line=dict(color=line_color), name=line_name
    ))
    
    # Punti Predetti
    fig.add_trace(go.Scatter(
        x=x_points, y=mean_prediction, mode='markers',
        marker=dict(color='blue', size=8, symbol='circle'), name=points_name
    ))
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Curvature [rad]",
        yaxis_title=y_title,
        hovermode="x unified", template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis_nticks=10 # <-- Aggiunto per griglia asse X più fitta
    )
    
    # 6. Linea Threshold
    if threshold_value > 0:
        fig.add_hline(
            y=threshold_value, line_dash="dash", line_color="orange",
            annotation_text=f"{threshold_prefix}: {threshold_value}%", 
            annotation_position="top left"
        )
        
    return fig

# ==============================================================================
# 4. CALLBACK
# ==============================================================================

@app.callback(
    [Output('dr-prediction-graph', 'figure'),
     Output('ar-prediction-graph', 'figure'),
     Output('ap-prediction-graph', 'figure')],
    
    [
        Input('slider-od', 'value'),
        Input('slider-wt', 'value'),
        Input('slider-al', 'value'),
        Input('slider-ep', 'value'),
        Input('slider-ym', 'value'),
        Input('slider-sy1', 'value'),
        Input('slider-sy2', 'value'),
        Input('slider-ep2', 'value'), 
        Input('slider-oval', 'value'),
        Input('slider-dr-line', 'value'),
        Input('slider-ar-line', 'value'), 
        Input('slider-curv', 'value')
    ]
)
def update_graphs(od, wt, al, ep, ym, sy1, sy2, ep2, oval, dr_line_value, ar_line_value, curv): 

    # --- Input Scalari Comuni per GPR (DR & AR) ---
    test_scalari_gpr = np.array([[od, wt, al, ep, ym, sy1, sy2, ep2, oval]])
    test_scalari_gpr_norm = scaler_scalari_gpr.transform(test_scalari_gpr)
    
    # Asse X (Curvature) comune per GPR DR e AR
    test_vettoriali_gpr_x = params_gpr_plot_axis[0].flatten() # Prende le curvature dal primo scenario caricato

    
    # === GRAFICO DR ===
    # Chiamata alla funzione helper
    fig_dr = create_gpr_figure(
        model_gpr=gaussian_process_dr,
        scaler_out=scaler_output_dr,
        test_scalari_norm=test_scalari_gpr_norm,
        x_points=test_vettoriali_gpr_x,
        threshold_value=dr_line_value,
        title="Diameter Reduction (DR) - GPR",
        y_title="Diameter Reduction [%]",
        line_color='green',
        fill_color='rgba(255, 127, 14, 0.25)',
        line_name='Predicted DR (PCHIP)',
        points_name='Predicted DR Points',
        ci_name='95% Confidence Interval (DR)',
        threshold_prefix='DR Threshold'
    )
    
    # === GRAFICO AR ===
    # Chiamata alla funzione helper
    fig_ar = create_gpr_figure(
        model_gpr=gaussian_process_ar,
        scaler_out=scaler_output_ar,
        test_scalari_norm=test_scalari_gpr_norm,
        x_points=test_vettoriali_gpr_x,
        threshold_value=ar_line_value,
        title="Area Reduction (AR) - GPR",
        y_title="Area Reduction [%]",
        line_color='purple',
        fill_color='rgba(31, 119, 180, 0.25)',
        line_name='Predicted AR (PCHIP)',
        points_name='Predicted AR Points',
        ci_name='95% Confidence Interval (AR)',
        threshold_prefix='AR Threshold'
    )

    # === GRAFICO AP (MLP) ===
    fig_ap = go.Figure()
    # Prepara input per MLP usando la curvatura dallo slider 'slider-curv'
    scalar_features_mlp = np.array([[
        od, wt, al, ep, ym, sy1, sy2, ep2, oval, curv # Usa 'curv' dallo slider
    ]], dtype=np.float32)
    n_abs = pipe_abs.shape[0]
    pipe_abs_vector = pipe_abs.reshape(-1, 1)
    scalar_matrix_mlp = np.tile(scalar_features_mlp, (n_abs, 1))
    X_mlp_long = np.hstack([scalar_matrix_mlp, pipe_abs_vector])
    X_mlp_scaled = scaler_mlp.transform(X_mlp_long)
    X_mlp_tensor = torch.tensor(X_mlp_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_mlp_tensor = model_mlp(X_mlp_tensor)

    y_pred_profile_original = y_pred_mlp_tensor.cpu().numpy().flatten()

    # Logica per estendere il profilo 
    len_original_abs = pipe_abs[-1] - pipe_abs[0]
    pipe_abs_extended = np.linspace(pipe_abs[0], pipe_abs[-1] + len_original_abs, num=2*n_abs)
    y_pred_profile_full = np.concatenate((y_pred_profile_original, y_pred_profile_original[::-1]))

    fig_ap.add_trace(go.Scatter(
        x=pipe_abs_extended,
        y=y_pred_profile_full,
        mode='lines',
        line=dict(color='blue'),
        name='Predicted Area Profile'
    ))

    # Calcolo A0 (AREA INTERNA) e soglia minima area se ar_line_value > 0
    a_min_threshold_int = None # Inizializza a None fuori dall'if
    if ar_line_value > 0:
        od_mm = od * 25.4
        r_ext_mm = od_mm / 2.0
        r_int_mm = r_ext_mm - wt 
        
        # Calcola l'area interna iniziale (A0) in mm²
        a0_int = np.pi * (r_ext_mm ** 2)
        # Calcola l'area minima ammessa in base alla soglia di AR impostata (%)
        a_min_threshold_int = a0_int * (1.0 - ar_line_value / 100.0)

        # Aggiungi linea orizzontale al grafico AP basata sull'area interna minima
        fig_ap.add_hline(
            y=a_min_threshold_int, # Usa il valore basato sull'area interna
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Min Area : {a_min_threshold_int:.2e} mm² (AR {ar_line_value}%)", # Testo annotazione aggiornato
            annotation_position="bottom left"
        )
        
    # Logica per impostare Y range (invariata)
    if y_pred_profile_full.size > 0:
        data_min_pred = np.min(y_pred_profile_full)
        data_max_pred = np.max(y_pred_profile_full)
        data_mean = np.mean(y_pred_profile_full)
        margin = abs(data_mean * 0.05)
        if margin < 1e-9:
            data_range = data_max_pred - data_min_pred
            margin = data_range * 0.05 if data_range > 1e-9 else 0.1
        final_y_lower = min(data_min_pred, data_mean) - margin
        final_y_upper = max(data_max_pred, data_mean) + margin

        # Considera anche la linea di soglia (basata su area interna) nel range Y, SE CALCOLATA
        if a_min_threshold_int is not None:
            final_y_lower = min(final_y_lower, a_min_threshold_int * 0.98)
            final_y_upper = max(final_y_upper, a_min_threshold_int * 1.02)

        fig_ap.update_layout(yaxis_range=[final_y_lower, final_y_upper])

    fig_ap.update_layout(
        title="Area Profile (AP) - MLP", 
        xaxis_title="Curvilinear Abscissa [mm]",
        yaxis_title="Area Profile [mm^2]",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig_ap.update_yaxes(tickformat='.2e')
    return fig_dr, fig_ar, fig_ap


# ==============================================================================
# Esecuzione dell'App
# ==============================================================================

if __name__ == '__main__':
    print(f"\n--- Server Dash pronto ---")
    print(f"Aprire http://127.0.0.1:8050/ nel browser")
    app.run(debug=False, port=8050) # Specificata porta

