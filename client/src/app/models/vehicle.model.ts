export class Vehicle{
    public id: number;
    public vehicle_id: string;
    public vehicle_class: string;
    public accuracy: string;
    public speed: string;
    public position: string;
    public state: string;
    public time: string;

    constructor(id: number,vehicle_id: string,vehicle_class: string,accuracy: string,speed: string,position: string ,state: string, time: string){
        this.id=id;
        this.vehicle_id=vehicle_id;
        this.vehicle_class=vehicle_class;
        this.accuracy=accuracy;
        this.speed=speed;
        this.position=position;
        this.state=state;
        this.time = time;
    }
}