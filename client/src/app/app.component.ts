import { Component } from '@angular/core';
import { ApiService } from './api.service';
import { Vehicle } from './models/vehicle.model';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  providers: [ApiService]
})
export class AppComponent {
  // vehicles: Vehicle[] = [];
  vehicles: Vehicle[] = [];
  data: any;
  constructor(private api: ApiService){
    this.getVehicles();
  }

  getVehicles = () => {
    this.api.getAllVehicles().subscribe(
      data => {
        this.vehicles = data;
        setTimeout(()=>{   
          $('#datatableexample').DataTable( {
            pagingType: 'full_numbers',
            pageLength: 5,
            processing: true,
            lengthMenu : [5, 10, 25]
        } );
        }, 1);
      },
      error => {
        console.log(error);
      }
    );
  }

  vehicleClicked = (vehicle: {id : any}) => {
    this.api.getOneVehicles(vehicle.id).subscribe(
      data => {
        console.log(data);
      },
      error => {
        console.log(error);
      }
    );
  }
  

  
}
