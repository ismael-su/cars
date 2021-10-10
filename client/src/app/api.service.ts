import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {


  baseurl = "http://127.0.0.1:8000";
  httpHeaders = new HttpHeaders({'Content-Type': 'application/json'});

  constructor(private http: HttpClient) { }

  getAllVehicles(): Observable<any>{
    return this.http.get(this.baseurl + '/api/vehicle/',
    {headers: this.httpHeaders})
  }

  getOneVehicles(id: any): Observable<any>{
    return this.http.get(this.baseurl + '/api/vehicle/' + id + '/',
    {headers: this.httpHeaders})
  }
  getdata(params: any, type: any): void {
    $.getJSON("")
  }
}
