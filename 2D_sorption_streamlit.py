import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
# ============================================================
# DomenicoSolver (2D version)
# ============================================================

class DomenicoSolver:
    DOMENICO_SOLUTION = 0
    CLEARYUNG_SOLUTION = 1
    DIFFERENCE_SOLUTION = 2
    UNDEFINED_SOLUTION = 3

    C0 = 1.0
    a_x = a_y = 0.0
    v_x = 0.0
    R = 1.0
    Y = 0.0
    lambda_ = 0.0

    lastT = 0.0

    solutionType = DOMENICO_SOLUTION

    AMAX = 170
    nmax = 60
    glIdx = 1

    zn = [
        [0.076526521133497, 0.22778585114164, 0.37370608871542, 0.51086700195082,
         0.63605368072651, 0.74633190646015, 0.83911697182221, 0.91223442825132,
         0.96397192727791, 0.99312859918509],

        [0.025959772301248, 0.077809333949536, 0.12944913539694, 0.18073996487342,
         0.23154355137603, 0.28172293742326, 0.33114284826845, 0.37967005657680,
         0.42717374158308, 0.47352584176171, 0.51860140005856, 0.56227890075394,
         0.60444059704851, 0.64497282848947, 0.68376632738135, 0.72071651335573,
         0.75572377530658, 0.78869373993226, 0.81953752616214, 0.84817198478593,
         0.87451992264690, 0.89851031081004, 0.92007847617762, 0.93916627611642,
         0.95572225583999, 0.96970178876505, 0.98106720175259, 0.98978789522222,
         0.99584052511884, 0.99921012322744],

        [0.0162767448496020, 0.0488129851360490, 0.0812974954644250, 0.1136958501106650,
         0.1459737146548960, 0.1780968823676180, 0.2100313104605670, 0.2417431561638400,
         0.2731988125910490, 0.3043649443544960, 0.3352085228926250, 0.3656968614723130,
         0.3957976498289080, 0.4254789884073000, 0.4547094221677430, 0.4834579739205960,
         0.5116941771546670, 0.5393881083243570, 0.5665104185613970, 0.5930323647775720,
         0.6189258401254680, 0.6441634037849670, 0.6687183100439160, 0.6925645366421710,
         0.7156768123489670, 0.7380306437444000, 0.7596023411766470, 0.7803690438674330,
         0.8003087441391400, 0.8194003107379310, 0.8376235112281870, 0.8549590334346010,
         0.8713885059092960, 0.8868945174024200, 0.9014606353158520, 0.9150714231208980,
         0.9277124567223080, 0.9393703397527550, 0.9500327177844270, 0.9596882914487420,
         0.9683268284632640, 0.9759391745851360, 0.9825172635630140, 0.9880541263296230,
         0.9925439003237620, 0.9959818429872090, 0.9983643758631810, 0.9996895038832300]
    ]

    wn = [
        [0.15275338713072, 0.14917298647260, 0.14209610931838, 0.13168863844917,
         0.11819453196152, 0.10193011981724, 0.083276741576705, 0.062672048334108,
         0.040601429800387, 0.017614007139152],

        [0.051907877631221, 0.051767943174910, 0.051488451500981, 0.051070156069855,
         0.050514184532509, 0.049822035690550, 0.048995575455757, 0.048037031819971,
         0.046948988848912, 0.045734379716114, 0.044396478795787, 0.042938892835935,
         0.041365551235585, 0.039680695452381, 0.037888867569243, 0.035994898051084,
         0.034003892724946, 0.031921219019296, 0.029752491500789, 0.027503556749925,
         0.025180477621521, 0.022789516943998, 0.020337120729457, 0.017829901014208,
         0.015274618596785, 0.012678166476816, 0.010047557182288, 0.0073899311633454,
         0.0047127299269535, 0.0020268119688737],

        [0.0325506144923630, 0.0325161187138680, 0.0324471637140640, 0.0323438225685750,
         0.0322062047940300, 0.0320344562319920, 0.0318287588944110, 0.0315893307707270,
         0.0313164255968610, 0.0310103325863130, 0.0306713761236690, 0.0302999154208270,
         0.0298963441363280, 0.0294610899581670, 0.0289946141505550, 0.0284974110650850,
         0.0279700076168480, 0.0274129627260290, 0.0268268667255910, 0.0262123407356720,
         0.0255700360053490, 0.0249006332224830, 0.0242048417923640, 0.0234833990859260,
         0.0227370696583290, 0.0219666444387440, 0.0211729398921910, 0.0203567971543330,
         0.0195190811401450, 0.0186606796274110, 0.0177825023160450, 0.0168854798642450,
         0.0159705629025620, 0.0150387210269940, 0.0140909417723140, 0.0131282295669610,
         0.0121516046710880, 0.0111621020998380, 0.0101607705350080, 0.0091486712307830,
         0.0081268769256980, 0.0070964707911530, 0.0060585455042350, 0.0050142027429270,
         0.0039645543384440, 0.0029107318179340, 0.0018539607889460, 0.0007967920655520]
    ]

    d_x = d_y = d_k = 0.0
    tmpVal1 = tmpVal2 = tmpVal3 = 0.0

    @staticmethod
    def get_input_conc():
        return DomenicoSolver.C0

    @staticmethod
    def get_parameters():
        # For cross-section use: pack everything needed, including t
        return [
            DomenicoSolver.C0,
            DomenicoSolver.a_x,
            DomenicoSolver.a_y,
            DomenicoSolver.v_x,
            DomenicoSolver.R,
            DomenicoSolver.Y,
            DomenicoSolver.lambda_,
            DomenicoSolver.lastT,
        ]

    @staticmethod
    def set_solver_method(method):
        if DomenicoSolver.DOMENICO_SOLUTION <= method < DomenicoSolver.UNDEFINED_SOLUTION:
            DomenicoSolver.solutionType = method
        else:
            DomenicoSolver.solutionType = DomenicoSolver.DOMENICO_SOLUTION

    @staticmethod
    def set_num_integration_points(n):
        if n == 0:
            DomenicoSolver.nmax = 20
            DomenicoSolver.glIdx = 0
        elif n == 1:
            DomenicoSolver.nmax = 60
            DomenicoSolver.glIdx = 1
        else:
            DomenicoSolver.nmax = 96
            DomenicoSolver.glIdx = 2

    @staticmethod
    def set_parameters(C0, ax, ay, vx, R, Y, lam):
        DomenicoSolver.C0 = C0
        DomenicoSolver.a_x = ax
        DomenicoSolver.a_y = ay
        DomenicoSolver.v_x = vx
        DomenicoSolver.R = R
        DomenicoSolver.Y = Y
        DomenicoSolver.lambda_ = lam

        DomenicoSolver.d_x = ax * vx
        DomenicoSolver.d_y = ay * vx
        DomenicoSolver.d_k = lam

        DomenicoSolver.tmpVal1 = math.sqrt(1.0 + 4 * lam * ax / vx) if vx != 0 else 1.0
        DomenicoSolver.tmpVal2 = (1.0 / (2.0 * ax)) * (1.0 - DomenicoSolver.tmpVal1) if ax != 0 else 0.0
        DomenicoSolver.tmpVal3 = 2.0 * math.sqrt(ax * vx / R) if R != 0 else 0.0

    @staticmethod
    def erfc(x):
        tmp = abs(x)
        if tmp > 3:
            tmp2 = tmp * tmp
            tmp4 = tmp2 * tmp2
            f1 = (1.0 - 0.5/tmp2 + 0.75/(tmp4 - 5.0/(6.0*tmp4*tmp2)))
            fun = f1 * math.exp(-tmp2) / (tmp * math.sqrt(math.pi))
            return fun if x >= 0 else 2.0 - fun
        t = 1.0 / (1.0 + 0.3275911 * tmp)
        poly = t * (0.254829592 + t * (-0.284496736 +
               t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        fun = poly * math.exp(-tmp * tmp)
        return fun if x >= 0 else 2.0 - fun

    @staticmethod
    def erf(x):
        return 1.0 - DomenicoSolver.erfc(x)

    @staticmethod
    def exerfc(a, b):
        xa = math.exp(a)
        if abs(a) > DomenicoSolver.AMAX and b <= 0.0:
            return 0.0
        if abs(b) < 1e-12:
            return xa
        c = a - b*b
        if abs(c) > DomenicoSolver.AMAX and b > 0.0:
            return 0.0
        if c < -DomenicoSolver.AMAX:
            if b < 0.0:
                return 2.0*xa
            return 0.0

        x = abs(b)
        if x > 3.0:
            y = 0.5641896/(x+0.5/(x+1.0/(x+1.5/(x+2.0/(x+2.5/(x+1.0))))))  # noqa
        else:
            t = 1.0/(1.0+0.3275911*x)
            y = t*(0.2548296 - t*(0.2844967 - t*(1.421414 - t*(1.453152 - 1.061405*t))))
        if b < 0.0:
            return 2.0*xa - y*math.exp(c)
        return y*math.exp(c)

    @staticmethod
    def cnrmli(x, y, t):
        y1 = -DomenicoSolver.Y/2.0
        y2 = DomenicoSolver.Y/2.0

        if t == 0.0:
            return 0.0
        if x == 0.0:
            if abs(y) < y2:
                return DomenicoSolver.C0
            elif abs(y) == y2:
                return 0.5 * DomenicoSolver.C0

        sum_ = 0.0
        tt = t**0.25
        dx = DomenicoSolver.d_x / DomenicoSolver.R if DomenicoSolver.R != 0 else 0.0
        dy = DomenicoSolver.d_y / DomenicoSolver.R if DomenicoSolver.R != 0 else 0.0
        vx = DomenicoSolver.v_x / DomenicoSolver.R if DomenicoSolver.R != 0 else 0.0

        if dx == 0 or dy == 0:
            return 0.0

        for i in range(DomenicoSolver.nmax // 2):
            wi = DomenicoSolver.wn[DomenicoSolver.glIdx][i]
            zi_base = DomenicoSolver.zn[DomenicoSolver.glIdx][i]
            for sign in (+1, -1):
                zi = tt * (zi_base * sign + 1.0) / 2.0
                zsq = zi * zi
                z4 = zsq * zsq

                xvt = x - vx * z4
                exp1 = -xvt * xvt / (4.0 * dx * z4) - DomenicoSolver.d_k * z4

                erfc1 = (y1 - y) / (2.0 * zsq * math.sqrt(dy))
                z1 = DomenicoSolver.exerfc(exp1, erfc1)

                erfc2 = (y2 - y) / (2.0 * zsq * math.sqrt(dy))
                z2 = DomenicoSolver.exerfc(exp1, erfc2)

                sum_ += (z1 - z2) * wi / (zi * zsq)

        sum_ *= tt / 2.0
        cn = sum_ * x / math.sqrt(math.pi * dx)
        return cn * DomenicoSolver.C0

    @staticmethod
    def C(x, y, t):
        DomenicoSolver.lastT = t

        domenico = 0.0
        cleary = 0.0

        if DomenicoSolver.solutionType in (DomenicoSolver.DOMENICO_SOLUTION,
                                           DomenicoSolver.DIFFERENCE_SOLUTION):

            if DomenicoSolver.tmpVal3 == 0 or t <= 0:
                domenico = 0.0
            else:
                tmp4 = (x - t * DomenicoSolver.v_x / DomenicoSolver.R *
                        DomenicoSolver.tmpVal1) / (DomenicoSolver.tmpVal3 * math.sqrt(t))

                tmp9 = 2.0 * math.sqrt(DomenicoSolver.a_y * x) if DomenicoSolver.a_y > 0 else 1e-12
                tmp5 = DomenicoSolver.erf((y + DomenicoSolver.Y / 2.0) / tmp9)
                tmp6 = DomenicoSolver.erf((y - DomenicoSolver.Y / 2.0) / tmp9)

                # 2D: vertical term collapses to 2
                tmp7 = 2.0
                tmp8 = 0.0

                domenico = (DomenicoSolver.C0 / 8.0 *
                            math.exp(x * DomenicoSolver.tmpVal2) *
                            DomenicoSolver.erfc(tmp4) *
                            (tmp5 - tmp6) * (tmp7 - tmp8))

        if DomenicoSolver.solutionType in (DomenicoSolver.CLEARYUNG_SOLUTION,
                                           DomenicoSolver.DIFFERENCE_SOLUTION):
            cleary = DomenicoSolver.cnrmli(x, y, t)

        if DomenicoSolver.solutionType == DomenicoSolver.DOMENICO_SOLUTION:
            return domenico
        if DomenicoSolver.solutionType == DomenicoSolver.CLEARYUNG_SOLUTION:
            return cleary
        if DomenicoSolver.solutionType == DomenicoSolver.DIFFERENCE_SOLUTION:
            return abs(domenico - cleary)
        return domenico


# ============================================================
# Streamlit helpers for "canvas" equivalents
# ============================================================

def compute_concentration_field(x_min, x_max, y_min, y_max, nx, ny, t_val):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_max, y_min, ny)  # top to bottom like canvas
    X, Y = np.meshgrid(xs, ys)
    C = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            C[j, i] = DomenicoSolver.C(float(X[j, i]), float(Y[j, i]), t_val)
    return xs, ys, C


def plot_concentration_field(xs, ys, C, C0):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        C / (C0 if C0 != 0 else 1.0),
        extent=[xs[0], xs[-1], ys[-1], ys[0]],
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="jet",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Concentration")
    return fig


def compute_cross_section(params, const_coord, type_, x_limits, y_limits, n_points=500):
    # params: [C0, ax, ay, vx, R, Y, lam, lastT]
    C0, ax, ay, vx, R, Y, lam, t_val = params

    DomenicoSolver.set_parameters(C0, ax, ay, vx, R, Y, lam)

    if type_ == "CONSTANT_Y":
        xs = np.linspace(x_limits[0], x_limits[1], n_points)
        conc = np.array([DomenicoSolver.C(x, const_coord, t_val) for x in xs])
        horiz_label = "X"
    else:
        xs = np.linspace(y_limits[0], y_limits[1], n_points)
        conc = np.array([DomenicoSolver.C(const_coord, y, t_val) for y in xs])
        horiz_label = "Y"

    return xs, conc, horiz_label


def plot_cross_sections(cs_sets, y_max):
    fig, ax = plt.subplots(figsize=(10, 4))
    for cs in cs_sets:
        ax.plot(cs["x"], cs["c"], label=cs["label"], color=cs["color"])
    ax.set_xlim(cs_sets[0]["x"].min(), cs_sets[0]["x"].max())
    ax.set_ylim(0, y_max)
    ax.set_xlabel(cs_sets[0]["horiz_label"])
    ax.set_ylabel("Concentration")
    ax.legend()
    return fig


def get_curve_color(idx):
    colors = ["#0000FF", "#FF0000", "#990099", "#006600", "#999900"]
    if idx < len(colors):
        return colors[idx]
    return "#000000"


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="2D Equilibrium Sorption with 1st Order Decay", layout="wide")

st.title("2D Equilibrium Sorption with 1st Order Decay")

# Session state for field and cross sections
if "field" not in st.session_state:
    st.session_state.field = None
if "params" not in st.session_state:
    st.session_state.params = None
if "cs_sets" not in st.session_state:
    st.session_state.cs_sets = []

# Layout: left controls, right plots
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Model Parameters")

    C0 = st.number_input("C0 (input concentration)", value=1.0, format="%.4f")
    ax = st.number_input("a_x (dispersivity in x)", value=0.1, format="%.4f")
    ay = st.number_input("a_y (dispersivity in y)", value=0.01, format="%.4f")
    vx = st.number_input("v_x (velocity)", value=1.0, format="%.4f")
    R = st.number_input("R (retardation factor)", value=1.0, format="%.4f")
    Y = st.number_input("Source width Y", value=1.0, format="%.4f")
    lam = st.number_input("lambda (decay)", value=0.0, format="%.4f")

    st.markdown("---")

    st.subheader("Domain & Time")

    x_min = st.number_input("x min", value=0.01, format="%.4f")
    x_max = st.number_input("x max", value=1.0, format="%.4f")
    y_min = st.number_input("y min", value=-1.0, format="%.4f")
    y_max = st.number_input("y max", value=1.0, format="%.4f")

    nx = st.number_input("Number of grid cells in x", min_value=10, max_value=400, value=100, step=10)
    ny = st.number_input("Number of grid cells in y", min_value=10, max_value=400, value=50, step=10)

    t_val = st.number_input("Time t", value=1.0, format="%.4f")

    st.markdown("---")

    st.subheader("Solution Options")

    sol_type = st.selectbox(
        "Solution type",
        ["Domenico", "Cleary-Yung", "Difference"],
        index=0
    )
    if sol_type == "Domenico":
        DomenicoSolver.set_solver_method(DomenicoSolver.DOMENICO_SOLUTION)
    elif sol_type == "Cleary-Yung":
        DomenicoSolver.set_solver_method(DomenicoSolver.CLEARYUNG_SOLUTION)
    else:
        DomenicoSolver.set_solver_method(DomenicoSolver.DIFFERENCE_SOLUTION)

    gl_choice = st.selectbox(
        "Gaussian integration points",
        ["20", "60", "96"],
        index=1
    )
    if gl_choice == "20":
        DomenicoSolver.set_num_integration_points(0)
    elif gl_choice == "60":
        DomenicoSolver.set_num_integration_points(1)
    else:
        DomenicoSolver.set_num_integration_points(2)

    st.markdown("---")

    compute_button = st.button("Compute Concentration Field")

with right_col:
    st.subheader("Concentration Field")

    if compute_button:
        DomenicoSolver.set_parameters(C0, ax, ay, vx, R, Y, lam)
        xs, ys, C = compute_concentration_field(x_min, x_max, y_min, y_max, nx, ny, t_val)
        st.session_state.field = {"xs": xs, "ys": ys, "C": C, "C0": C0}
        DomenicoSolver.lastT = t_val
        st.session_state.params = DomenicoSolver.get_parameters()
        st.session_state.cs_sets = []  # reset cross sections when recomputing

    if st.session_state.field is not None:
        field = st.session_state.field
        fig_field = plot_concentration_field(field["xs"], field["ys"], field["C"], field["C0"])
        st.pyplot(fig_field)
    else:
        st.info("Click **Compute Concentration Field** to generate the plume.")

    st.markdown("---")
    st.subheader("Cross Sections")

    if st.session_state.field is not None and st.session_state.params is not None:
        cs_type = st.radio("Cross-section type", ["Constant Y", "Constant X"], horizontal=True)
        if cs_type == "Constant Y":
            const_coord = st.slider("Y coordinate", float(y_min), float(y_max), 0.0)
            type_key = "CONSTANT_Y"
        else:
            const_coord = st.slider("X coordinate", float(x_min), float(x_max), 1.0)
            type_key = "CONSTANT_X"

        add_cs = st.button("Add Cross Section")

        if add_cs:
            xs_cs, conc_cs, horiz_label = compute_cross_section(
                st.session_state.params,
                const_coord,
                type_key,
                (x_min, x_max),
                (y_min, y_max),
                n_points=500
            )
            idx = len(st.session_state.cs_sets)
            label = f"{'Y' if type_key=='CONSTANT_Y' else 'X'} = {const_coord:.3f}"
            color = get_curve_color(idx)
            st.session_state.cs_sets.append({
                "x": xs_cs,
                "c": conc_cs,
                "label": label,
                "color": color,
                "horiz_label": horiz_label
            })

        if len(st.session_state.cs_sets) > 0:
            fig_cs = plot_cross_sections(st.session_state.cs_sets, y_max=C0)
            st.pyplot(fig_cs)
        else:
            st.info("Add one or more cross sections to see the curves.")
    else:
        st.info("Cross sections become available after computing the concentration field.")

st.markdown("---")

# Data views
if st.session_state.field is not None:
    field = st.session_state.field
    xs, ys, C = field["xs"], field["ys"], field["C"]

    with st.expander("Concentration Grid Data"):
        xs = field["xs"]
        ys = field["ys"]
        C = field["C"]
    
        # Build labeled matrix
        # First row: ["y\\x", x1, x2, x3, ...]
        header = ["y\\x"] + [f"{x:.4f}" for x in xs]
    
        # Each subsequent row: [y_j, C[j,0], C[j,1], ...]
        table = []
        for j, y in enumerate(ys):
            row = [f"{y:.4f}"] + [f"{C[j,i]:.4f}" for i in range(len(xs))]
            table.append(row)
    
        # Display as a dataframe
        st.dataframe(
            pd.DataFrame(table, columns=header),
            use_container_width=True
        )

if len(st.session_state.cs_sets) > 0:
    with st.expander("Cross Section Data"):
        for cs in st.session_state.cs_sets:
            st.write(f"**{cs['label']}**")
            st.write("Coordinate, Concentration")
            st.write(np.column_stack([cs["x"], cs["c"]]))



