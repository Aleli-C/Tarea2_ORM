from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuración de la base de datos (SQLite en este caso)
# ---------------------------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./flights.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Modelo ORM para la tabla "flights"
# ---------------------------------------------------------------------------
class Flight(Base):
    __tablename__ = "flights"
    id = Column(Integer, primary_key=True, index=True)
    flight_number = Column(String, unique=True, index=True)
    origin = Column(String)
    destination = Column(String)
    emergency = Column(Boolean, default=False)
    scheduled_time = Column(DateTime, default=datetime.utcnow)
    # Campo para mantener la posición en la lista (opcional para sincronización)
    position = Column(Integer)

# Crear las tablas en la base de datos (si aún no existen)
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Implementación de la Lista Doblemente-enlazada
# Cada nodo contendrá una instancia de "Flight"
# ---------------------------------------------------------------------------

class FlightNode:
    def __init__(self, flight: Flight):
        self.flight = flight
        self.prev: Optional['FlightNode'] = None
        self.next: Optional['FlightNode'] = None

class FlightDoublyLinkedList:
    def __init__(self):
        self.head: Optional[FlightNode] = None
        self.tail: Optional[FlightNode] = None
        self.size = 0

    def insertar_al_frente(self, flight: Flight):
        """Inserta un vuelo al comienzo de la lista (usado para emergencias)."""
        node = FlightNode(flight)
        if self.head is None:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
        self.size += 1

    def insertar_al_final(self, flight: Flight):
        """Inserta un vuelo al final de la lista (vuelos regulares)."""
        node = FlightNode(flight)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        self.size += 1

    def obtener_primero(self) -> Optional[Flight]:
        """Devuelve el primer vuelo (sin removerlo)"""
        return self.head.flight if self.head else None

    def obtener_ultimo(self) -> Optional[Flight]:
        """Devuelve el último vuelo (sin removerlo)"""
        return self.tail.flight if self.tail else None

    def longitud(self) -> int:
        """Retorna la cantidad total de vuelos en la lista."""
        return self.size

    def insertar_en_posicion(self, flight: Flight, posicion: int):
        """
        Inserta un vuelo en una posición específica.
        Si la posición es 0 o igual a la longitud, utiliza los métodos
        predefinidos para insertar al frente o al final.
        """
        if posicion < 0 or posicion > self.size:
            raise IndexError("Posición fuera del rango permitido")
        if posicion == 0:
            self.insertar_al_frente(flight)
        elif posicion == self.size:
            self.insertar_al_final(flight)
        else:
            node = FlightNode(flight)
            current = self.head
            for _ in range(posicion):
                current = current.next
            prev_node = current.prev
            prev_node.next = node
            node.prev = prev_node
            node.next = current
            current.prev = node
            self.size += 1

    def extraer_de_posicion(self, posicion: int) -> Flight:
        """
        Remueve y retorna el vuelo en la posición dada (útil para cancelaciones).
        """
        if posicion < 0 or posicion >= self.size:
            raise IndexError("Posición fuera del rango permitido")
        # Caso: remover el primer elemento
        if posicion == 0:
            node = self.head
            self.head = node.next
            if self.head:
                self.head.prev = None
            if node == self.tail:
                self.tail = None
            self.size -= 1
            return node.flight
        current = self.head
        for _ in range(posicion):
            current = current.next
        # Ajustar enlaces de los nodos vecinos
        if current.prev:
            current.prev.next = current.next
        if current.next:
            current.next.prev = current.prev
        if current == self.tail:
            self.tail = current.prev
        self.size -= 1
        return current.flight

# Instancia global de la lista de vuelos en memoria.
flight_list = FlightDoublyLinkedList()

# ---------------------------------------------------------------------------
# Mecanismo simple de Undo/Redo
# Utiliza dos pilas (stacks) para almacenar acciones.
# Cada acción es un diccionario que almacena:
#   - type: "insert" o "delete"
#   - flight: los datos del vuelo (como objeto Flight)
#   - position: la posición en la lista donde se realizó la acción
# ---------------------------------------------------------------------------
undo_stack = []
redo_stack = []

def record_undo(action: dict):
    """Registra una acción en la pila de undo y limpia la pila de redo."""
    undo_stack.append(action)
    redo_stack.clear()
# ---------------------------------------------------------------------------
# Esquemas Pydantic para validación y serialización
# ---------------------------------------------------------------------------
class FlightBase(BaseModel):
    flight_number: str
    origin: str
    destination: str
    emergency: bool = False
    scheduled_time: Optional[datetime] = None

class FlightCreate(FlightBase):
    pass

class FlightOut(FlightBase):
    id: int
    position: Optional[int]

    class Config:
        from_attributes = True

# ---------------------------------------------------------------------------
# Dependencia para obtener la sesión de la BD
# ---------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Instanciación de la aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Gestión de Vuelos - Torre de Control")

# ---------------------------------------------------------------------------
# Endpoints CRUD y manejo de la lista de vuelos
# ---------------------------------------------------------------------------

@app.post("/flights/", response_model=FlightOut)
def create_flight(flight: FlightCreate, db: Session = Depends(get_db)):
    """
    Inserta un vuelo. Se decide:
      - Si el vuelo es de emergencia, se inserta al frente.
      - Si es regular, se inserta al final.
    Se persiste en la base de datos y se agrega a la lista enlazada.
    """
    # Crear instancia del vuelo con datos y asignar posición provisional
    pos_actual = flight_list.longitud()
    db_flight = Flight(
        flight_number=flight.flight_number,
        origin=flight.origin,
        destination=flight.destination,
        emergency=flight.emergency,
        scheduled_time=flight.scheduled_time or datetime.utcnow(),
        position=pos_actual  # posición provisional
    )
    db.add(db_flight)
    db.commit()
    db.refresh(db_flight)
    
    # Inserción en la lista doblemente enlazada según prioridad
    if db_flight.emergency:
        flight_list.insertar_al_frente(db_flight)
        pos_insercion = 0
    else:
        flight_list.insertar_al_final(db_flight)
        pos_insercion = flight_list.longitud() - 1

    # Registrar la acción para poder deshacerla
    record_undo({"type": "insert", "flight": db_flight, "position": pos_insercion})
    return db_flight

@app.post("/flights/{position}", response_model=FlightOut)
def insert_flight_at_position(position: int, flight: FlightCreate, db: Session = Depends(get_db)):
    """
    Inserta un vuelo en la posición indicada de la lista.
    Útil para reorganizaciones o asignaciones especiales.
    """
    if position < 0 or position > flight_list.longitud():
        raise HTTPException(status_code=400, detail="Posición fuera de rango")
    
    db_flight = Flight(
        flight_number=flight.flight_number,
        origin=flight.origin,
        destination=flight.destination,
        emergency=flight.emergency,
        scheduled_time=flight.scheduled_time or datetime.utcnow(),
        position=position
    )
    db.add(db_flight)
    db.commit()
    db.refresh(db_flight)
    
    try:
        flight_list.insertar_en_posicion(db_flight, position)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    record_undo({"type": "insert", "flight": db_flight, "position": position})
    return db_flight

@app.delete("/flights/{position}", response_model=FlightOut)
def delete_flight_at_position(position: int, db: Session = Depends(get_db)):
    """
    Extrae (elimina) el vuelo que se encuentre en la posición dada de la lista,
    lo que simula la cancelación de un vuelo.
    """
    if flight_list.longitud() == 0:
        raise HTTPException(status_code=404, detail="No hay vuelos en la lista")
    try:
        # Extraer vuelo de la lista en memoria
        flight_removed = flight_list.extraer_de_posicion(position)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Eliminar el registro correspondiente de la base de datos
    db_flight = db.query(Flight).filter(Flight.id == flight_removed.id).first()
    if db_flight is None:
        raise HTTPException(status_code=404, detail="Vuelo no encontrado en la base de datos")
    db.delete(db_flight)
    db.commit()
    
    # Registrar acción de eliminación para poder deshacerla
    record_undo({"type": "delete", "flight": flight_removed, "position": position})
    return flight_removed

@app.get("/flights/first", response_model=FlightOut)
def get_first_flight():
    """Obtiene el primer vuelo (el de mayor prioridad en emergencias o el primero programado)."""
    flight = flight_list.obtener_primero()
    if flight is None:
        raise HTTPException(status_code=404, detail="No hay vuelos disponibles")
    return flight

@app.get("/flights/last", response_model=FlightOut)
def get_last_flight():
    """Obtiene el último vuelo de la lista."""
    flight = flight_list.obtener_ultimo()
    if flight is None:
        raise HTTPException(status_code=404, detail="No hay vuelos disponibles")
    return flight

@app.get("/flights/", response_model=List[FlightOut])
def get_all_flights():
    """Recorre la lista para retornar todos los vuelos en orden."""
    flights = []
    current = flight_list.head
    while current:
        flights.append(current.flight)
        current = current.next
    return flights

# ---------------------------------------------------------------------------
# Endpoints para Undo/Redo
# ---------------------------------------------------------------------------

@app.post("/undo")
def undo_action(db: Session = Depends(get_db)):
    """
    Deshace la última acción registrada (inserción o eliminación).
    El mecanismo se basa en una pila de acciones.
    """
    if not undo_stack:
        raise HTTPException(status_code=400, detail="No hay acciones para deshacer")
    action = undo_stack.pop()

    if action["type"] == "insert":
        # Deshacer una inserción: extraer y eliminar el vuelo
        pos = action["position"]
        try:
            flight_to_remove = flight_list.extraer_de_posicion(pos)
        except IndexError as e:
            raise HTTPException(status_code=400, detail=str(e))
        db_flight = db.query(Flight).filter(Flight.id == flight_to_remove.id).first()
        if db_flight:
            db.delete(db_flight)
            db.commit()
        # Registrar en redo la acción inversa
        redo_stack.append({"type": "delete", "flight": flight_to_remove, "position": pos})
        return {"detail": f"Deshecha la inserción del vuelo {flight_to_remove.flight_number}"}

    elif action["type"] == "delete":
        # Deshacer una eliminación: reinserta el vuelo en la posición indicada
        flight_data = action["flight"]
        pos = action["position"]
        new_flight = Flight(
            flight_number=flight_data.flight_number,
            origin=flight_data.origin,
            destination=flight_data.destination,
            emergency=flight_data.emergency,
            scheduled_time=flight_data.scheduled_time,
            position=pos
        )
        db.add(new_flight)
        db.commit()
        db.refresh(new_flight)
        flight_list.insertar_en_posicion(new_flight, pos)
        redo_stack.append({"type": "insert", "flight": new_flight, "position": pos})
        return {"detail": f"Deshecha la cancelación del vuelo {flight_data.flight_number}"}

    else:
        raise HTTPException(status_code=400, detail="Tipo de acción no reconocida")

@app.post("/redo")
def redo_action(db: Session = Depends(get_db)):
    """
    Rehace la última acción deshecha.
    Se utiliza una pila de redo para almacenar las acciones revertidas.
    """
    if not redo_stack:
        raise HTTPException(status_code=400, detail="No hay acciones para rehacer")
    action = redo_stack.pop()

    if action["type"] == "delete":
        # Rehacer una eliminación: extraer y eliminar nuevamente el vuelo
        pos = action["position"]
        try:
            flight_removed = flight_list.extraer_de_posicion(pos)
        except IndexError as e:
            raise HTTPException(status_code=400, detail=str(e))
        db_flight = db.query(Flight).filter(Flight.id == flight_removed.id).first()
        if db_flight:
            db.delete(db_flight)
            db.commit()
        undo_stack.append({"type": "insert", "flight": flight_removed, "position": pos})
        return {"detail": f"Rehecha la cancelación del vuelo {flight_removed.flight_number}"}

    elif action["type"] == "insert":
        # Rehacer una inserción: reinserta el vuelo
        pos = action["position"]
        flight_data = action["flight"]
        new_flight = Flight(
            flight_number=flight_data.flight_number,
            origin=flight_data.origin,
            destination=flight_data.destination,
            emergency=flight_data.emergency,
            scheduled_time=flight_data.scheduled_time,
            position=pos
        )
        db.add(new_flight)
        db.commit()
        db.refresh(new_flight)
        flight_list.insertar_en_posicion(new_flight, pos)
        undo_stack.append({"type": "delete", "flight": new_flight, "position": pos})
        return {"detail": f"Rehecha la inserción del vuelo {new_flight.flight_number}"}

    else:
        raise HTTPException(status_code=400, detail="Tipo de acción no reconocida")
